"""CPU contract test of production Boltz1 resident adapter and daemon."""
import json
from pathlib import Path

import pytest


def test_boltz1_resident_loads_once_and_fences_errors(tmp_path, monkeypatch):
    fake = tmp_path / 'boltz_fake.py'
    fake.write_text('''\
import json, os
from pathlib import Path

def install():
    from astrolabe.evaluators.wrappers import boltz as b
    class Model:
        @classmethod
        def load_from_checkpoint(cls, checkpoint, **kw):
            with open(os.environ['BOLTZ_TEST_LOADS'], 'a') as f:
                f.write(json.dumps({'checkpoint':str(checkpoint),'predict_args':kw['predict_args']})+'\\n')
            return cls()
        def eval(self): return self
    class Manifest:
        @staticmethod
        def load(path): return {'path': str(path)}
    class Writer:
        def __init__(self, **kwargs): pass
    class DataModule:
        def __init__(self, **kwargs): pass
    class Trainer:
        def __init__(self, **kwargs): self.root = Path(kwargs['default_root_dir'])
        def predict(self, model, **kwargs):
            value = json.loads((self.root/'processed'/'case.json').read_text())
            if value.get('fail'): raise RuntimeError('fixture prediction failure')
            out = self.root/'predictions'/'case'; out.mkdir(parents=True)
            (out/'result.json').write_text(json.dumps(value))
    def check_inputs(path, out, override): return [path]
    def process_inputs(*, data, out_dir, **kwargs):
        target = Path(out_dir)/'processed'; target.mkdir()
        (target/'manifest.json').write_text('{}')
        (target/'case.json').write_text(Path(data[0]).read_text())
    b.Boltz1, b.Manifest, b.BoltzWriter = Model, Manifest, Writer
    b.BoltzInferenceDataModule, b.Trainer = DataModule, Trainer
    b.check_inputs, b.process_inputs = check_inputs, process_inputs
''')
    (tmp_path / 'sitecustomize.py').write_text('from boltz_fake import install\ninstall()\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + ':' + __import__('os').environ.get('PYTHONPATH', ''))
    monkeypatch.setenv('BOLTZ_TEST_LOADS', str(tmp_path / 'loads'))
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    from boltz_fake import install
    install()
    from astrolabe.evaluators.boltz_resident import Boltz1Resident
    from astrolabe.runtime.client import WorkerError
    cache = tmp_path / 'cache'; cache.mkdir()
    (cache / 'ccd.pkl').write_bytes(b'ccd')
    (cache / 'boltz1_conf.ckpt').write_bytes(b'model')
    config = {'accelerator':'cpu', 'cache':str(cache), 'checkpoint':None, 'devices':1,
              'diffusion_samples':2, 'num_workers':0, 'output_format':'mmcif',
              'recycling_steps':4, 'sampling_steps':17, 'write_full_pae':True,
              'write_full_pde':False, 'use_msa_server':False, 'msa_server_url':None,
              'msa_pairing_strategy':'greedy'}
    worker = Boltz1Resident(config)
    try:
        assert worker._client.hello['load_count'] == 1
        for i in range(2):
            case = {'sequence': 'A' * (i + 1), 'constraints': [{'chain': 'A', 'residue': i + 1}], 'seed': i}
            source = tmp_path / f'{i}.yaml'; source.write_text(json.dumps(case))
            destination = tmp_path / f'output-{i}'
            data, request = worker.prepare_request(source, destination)
            worker.run_request(data, request, destination)
            assert json.loads((destination/'predictions'/'case'/'result.json').read_text()) == case
            assert not request.exists()
        loads = [json.loads(line) for line in (tmp_path/'loads').read_text().splitlines()]
        assert len(loads) == 1
        assert loads[0]['predict_args']['recycling_steps'] == 4
        assert loads[0]['predict_args']['sampling_steps'] == 17
        assert loads[0]['predict_args']['diffusion_samples'] == 2
        source = tmp_path / 'fail.yaml'; source.write_text('{"fail": true}')
        data, request = worker.prepare_request(source, tmp_path/'failed')
        with pytest.raises(WorkerError):
            worker.run_request(data, request, tmp_path/'failed')
        data, request = worker.prepare_request(source, tmp_path/'retry')
        with pytest.raises(WorkerError):
            worker.run_request(data, request, tmp_path/'retry')
        assert len((tmp_path/'loads').read_text().splitlines()) == 1
    finally:
        worker.close()
