"""RF3 native post-init RNG semantics through real production JSONL transport."""
import json
from pathlib import Path
import sys
import pytest

torch = pytest.importorskip('torch')


def test_rf3_common_worker_exact_config_reuse_and_native_rng_reset(tmp_path, monkeypatch):
    source = tmp_path / 'foundry'
    package = source / 'models' / 'rf3' / 'src' / 'rf3'
    for path in (package, package / '_core', package / '_core' / 'inference_engines', package / 'inference_engines'):
        path.mkdir(parents=True, exist_ok=True)
        (path / '__init__.py').write_text('')
    (package / '_core' / 'inference_engines' / 'base.py').write_text('import torch\ndef load_checkpoint(path): return torch.load(path, map_location="cpu")\n')
    (package / 'inference_engines' / 'rf3.py').write_text('''\
import json, random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from rf3._core.inference_engines import base
class RF3InferenceEngine:
    def __init__(self, ckpt_path, seed=None, **kwargs):
        self.ckpt_path, self.initialized_, self.seed = ckpt_path, False, seed
        if seed is not None:
            torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    def initialize(self):
        if self.initialized_: return
        weights = base.load_checkpoint(self.ckpt_path)
        model = torch.nn.Linear(1, 1)
        model.load_state_dict(weights)
        self.trainer = SimpleNamespace(state={'model': model})
        self.pipeline = object()
        self.initialized_ = True
    def run(self, *, inputs, out_dir, **kwargs):
        self.initialize()
        name = json.loads(Path(inputs).read_text())['name']
        out_dir = Path(out_dir)
        (out_dir / (name + '_model.cif')).write_text('data_model')
        values = {'torch': torch.rand(3).tolist(), 'python': random.random(), 'numpy': float(np.random.random())}
        (out_dir / (name + '_summary_confidences.json')).write_text(json.dumps(values))
''')
    checkpoint = tmp_path / 'rf3.pt'
    torch.save(torch.nn.Linear(1, 1).state_dict(), checkpoint)
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    from astrolabe.evaluators.wrappers.rf3 import RF3RunnerService, RF3RunnerRequest
    service = RF3RunnerService({'execution': 'daemon', 'repo_path': str(source), 'ckpt_path': str(checkpoint), 'python_bin': sys.executable})
    results = []
    try:
        for index, seed in enumerate((13, 13, 19)):
            path = tmp_path / f'input-{index}.json'; path.write_text(json.dumps({'name': f'job-{index}'}))
            request = RF3RunnerRequest(f'science-{index}', path, tmp_path / f'output-{index}', [], [], {'execution': 'daemon', 'seed': seed})
            result = service.submit(request)
            assert result.status == 'success', result.error
            results.append(result.summary_confidences)
        assert results[0] == results[1] and results[0] != results[2]
        assert len(service._resident_cache) == 2
        workers = list(service._resident_cache.values())
        processes = [worker._client.process for worker in workers]
        assert all(worker._client.lifecycle['load_count'] == 1 for worker in workers)
        assert all(worker._client.lifecycle['gpu_resident'] is False for worker in workers)
    finally:
        service.close()
    assert all(process.poll() is not None for process in processes)
    service.close()
