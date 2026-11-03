"""Production RhoFold worker with a tiny CPU model; no GPU certification."""
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')


def test_rhofold_native_draws_device_postprocessing_and_weight_reuse(tmp_path, monkeypatch):
    source = tmp_path / 'source' / 'rhofold'; source.mkdir(parents=True)
    (source / '__init__.py').write_text('')
    (source / 'config.py').write_text('rhofold_config = {}\n')
    (source / 'rhofold.py').write_text('''\
import os
from pathlib import Path
import torch
class RhoFold(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(1))
    def forward(self, tokens, rna_fm_tokens, seq):
        if seq == 'FAIL': raise RuntimeError('fixture forward failure')
        n = len(seq)
        value = torch.rand(1, 2, n, n) + self.weight
        return [{'ss': value[:, :1], 'n': value, 'p': value + 1, 'c4_': value + 2,
                 'plddt': torch.rand(1, n), 'cord_tns_pred': [torch.rand(1, n, 3)]}]
''')
    for name in list(sys.modules):
        if name == 'rhofold' or name.startswith('rhofold.'):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(source.parent))
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    checkpoint = tmp_path / 'model.pt'
    torch.save({'model': {'weight': torch.tensor([2.])}}, checkpoint)
    from astrolabe.evaluators.rhofold_resident import RhoFoldResident
    from rhofold.rhofold import RhoFold
    torch.manual_seed(89)
    initial = torch.get_rng_state().clone()
    direct = RhoFold({}); direct.load_state_dict(torch.load(checkpoint)['model']); direct.eval()
    inputs = [{'A': {'seq': 'ACG', 'tokens': torch.zeros((1, 1, 3)), 'rna_fm_tokens': torch.zeros((1, 5))}},
              {'A': {'seq': 'UU', 'tokens': torch.zeros((1, 1, 2)), 'rna_fm_tokens': torch.zeros((1, 4))}}]
    expected = []
    for item in inputs:
        feat = item['A']
        with torch.no_grad():
            value = direct(**{'tokens': feat['tokens'], 'rna_fm_tokens': feat['rna_fm_tokens'], 'seq': feat['seq']})[-1]
            expected.append({'ss_prob_map': torch.sigmoid(value['ss'][0, 0]).numpy(),
                             'dist_n': torch.softmax(value['n'].squeeze(0), dim=0).numpy(),
                             'plddt_cpu': value['plddt'][0].numpy()})
    final_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial)
    worker = RhoFoldResident({'device': 'cpu', 'checkpoint': str(checkpoint), 'relax_steps': 0})
    try:
        process = worker._client.process
        for item, reference in zip(inputs, expected):
            observed = worker.forward(item)['A']
            for key, value in reference.items():
                np.testing.assert_array_equal(observed[key], value)
        assert torch.equal(torch.get_rng_state(), final_rng)
        assert worker._client.lifecycle['load_count'] == 1
        assert worker._client.lifecycle['weights_loaded'] is True
        assert worker._client.lifecycle['gpu_resident'] is False
        assert worker._client.lifecycle['resident_bytes'] == 0
    finally:
        worker.close()
    assert process.poll() is not None
