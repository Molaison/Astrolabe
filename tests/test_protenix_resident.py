"""End-to-end direct JSONL handler with a fake runner and real torch RNG."""
import json
import random
import numpy as np
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf


def test_resident_trajectory_rng_and_dynamic_config(tmp_path, monkeypatch):
    import astrolabe.evaluators.wrappers.protenix as wrapper
    from astrolabe.evaluators.protenix_resident import ProtenixResident

    source = tmp_path / 'source'; (source / 'protenix').mkdir(parents=True); (source / 'configs').mkdir()
    (source / 'protenix' / '__init__.py').write_text('')
    (source / 'configs' / '__init__.py').write_text('')
    for name in ('checkpoint.pt', 'ccd.cif', 'ccd.pkl'):
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setenv('GH_PROTENIX_CCD', str(tmp_path / 'ccd.cif'))
    monkeypatch.setenv('GH_PROTENIX_CCD_RDKIT', str(tmp_path / 'ccd.pkl'))
    monkeypatch.delenv('ASTROLABE_EXECUTION_ENDPOINT', raising=False)
    monkeypatch.delenv('ASTROLABE_EXECUTION_TOKEN', raising=False)
    monkeypatch.setattr(wrapper, 'configure_protenix_source', lambda _: None)
    cfg = OmegaConf.create({'model_name': 'protenix-v2', 'seed': 17, 'load_checkpoint_path': str(tmp_path / 'checkpoint.pt'),
                            'dump_dir': str(tmp_path / 'outputs'), 'need_atom_confidence': False,
                            'sorted_by_ranking_score': True})
    settings = SimpleNamespace(seed=17, model_name='protenix-v2', load_checkpoint_path=str(tmp_path / 'checkpoint.pt'), dump_dir=str(tmp_path / 'outputs'), need_atom_confidence=False,
                               sorted_by_ranking_score=True,
                               skip_amp=SimpleNamespace(confidence_head=True, sample_diffusion=True))
    monkeypatch.setattr(wrapper, '_normalize_protenix_configs', lambda _: settings)
    monkeypatch.setattr(wrapper, 'DataDumper', lambda **kwargs: SimpleNamespace(dump=lambda **kw: None))
    hook = tmp_path / 'sitecustomize.py'
    hook.write_text('''\
from types import SimpleNamespace
from pathlib import Path
import random
import numpy as np
import torch
import astrolabe.evaluators.wrappers.protenix as p
p._ensure_protenix_ccd_cache_files_exist = lambda: None
p._lazy_import_protenix = lambda: None
def normalize(cfg):
    return SimpleNamespace(seed=cfg.seed, model_name=cfg.model_name, load_checkpoint_path=cfg.load_checkpoint_path, dump_dir=cfg.dump_dir, dtype='fp32',
                           need_atom_confidence=False, sorted_by_ranking_score=True,
                           skip_amp=SimpleNamespace(confidence_head=True, sample_diffusion=True))
p._normalize_protenix_configs = normalize
p.ProtenixRunner.init_env = lambda self: setattr(self, 'device', torch.device('cpu'))
class Model:
    def __call__(self, input_feature_dict, **kwargs):
        return {'random': torch.rand(1), 'python': random.random(), 'numpy': float(np.random.random()), 'token': input_feature_dict['token'], 'bf16': input_feature_dict['bf16']}, None, None
p.ProtenixRunner.init_model = lambda self: setattr(self, 'model', Model())
def load(self):
    self._weight_load_count += 1
    torch.rand(1); random.random(); np.random.random()
    with open(%r, 'a') as out: out.write('load\\n')
p.ProtenixRunner.load_checkpoint = load
p.to_device = lambda data, device: data
''' % str(tmp_path / 'loads.txt'))
    monkeypatch.syspath_prepend(str(tmp_path))
    torch.manual_seed(81)
    random.seed(81)
    np.random.seed(81)
    python_gen = random.Random(81)
    numpy_gen = np.random.RandomState(81)
    python_gen.random(); numpy_gen.random()
    expected_python = [python_gen.random(), python_gen.random(), python_gen.random()]
    expected_numpy = [numpy_gen.random(), numpy_gen.random(), numpy_gen.random()]
    gen = torch.Generator().manual_seed(81)
    torch.rand(1, generator=gen)  # same load-time draw as fake model
    first_expected = torch.rand(1, generator=gen)
    between_expected = torch.rand(1, generator=gen)
    second_expected = torch.rand(1, generator=gen)
    resident = ProtenixResident(cfg, source_dir=source)
    try:
        for index, n in enumerate((10, 3000)):
            settings.skip_amp.confidence_head = n <= 2560
            settings.skip_amp.sample_diffusion = n <= 3840
            resident.update_model_configs(settings)
            x = torch.tensor([1.25, -4], dtype=torch.bfloat16)
            result = resident.predict({'N_token': torch.tensor(n), 'input_feature_dict': {'bf16': x, 'token': torch.tensor(n)}})
            assert torch.equal(result['random'], first_expected if index == 0 else second_expected)
            assert result['python'] == expected_python[index * 2]
            assert result['numpy'] == expected_numpy[index * 2]
            assert result['token'].item() == n and result['bf16'].dtype == torch.bfloat16
            assert torch.equal(result['bf16'], x)
            if index == 0:
                assert torch.equal(torch.rand(1), between_expected)
                assert random.random() == expected_python[1]
                assert np.random.random() == expected_numpy[1]
        assert resident._client.hello['load_count'] == 1
        assert (tmp_path / 'loads.txt').read_text().splitlines() == ['load']
        from astrolabe.runtime.client import WorkerError
        settings.seed = 18
        resident.update_model_configs(settings)
        old_rng = torch.get_rng_state().clone()
        with pytest.raises(WorkerError, match='settings mismatch'):
            resident.predict({'N_token': torch.tensor(10), 'input_feature_dict': {'bf16': x, 'token': torch.tensor(10)}})
        assert torch.equal(torch.get_rng_state(), old_rng)
    finally:
        resident.close()


def test_resident_declares_shared_resource_contract():
    from astrolabe.evaluators import protenix_resident
    assert protenix_resident.BROKER_SHARED_ROOT
    assert protenix_resident.BROKER_ADMITTED_PROBE


@pytest.mark.parametrize('model_name', ['protenix_base_default_v0.5.0', 'protenix_base_constraint_v0.5.0', 'protenix_base_default_v1.0.0', 'protenix-v2'])
def test_factory_preserves_exact_variant_without_parent_model_load(tmp_path, monkeypatch, model_name):
    from astrolabe.core import global_config
    from astrolabe.evaluators.utils import construct_evaluators
    from astrolabe.evaluators.protenix_resident import ProtenixResident
    captured = []
    def init(self, conf, **kwargs):
        captured.append((OmegaConf.to_container(conf, resolve=True), kwargs))
    monkeypatch.setattr(ProtenixResident, '__init__', init)
    monkeypatch.setattr('astrolabe.evaluators.wrappers.protenix.ProtenixRunner', lambda *_: pytest.fail('parent loaded model'))
    global_config.reset()
    global_config.init_const('out_dir', tmp_path)
    global_config.init_const('protenix_constraint', None)
    global_config.init_const('binding_site', None)
    global_config.init_const('dynamic_binding_site_enabled', False)
    config = OmegaConf.create({'evaluator_terms': {'p': {'_type': 'protenix', 'execution': 'daemon', 'source_dir': str(tmp_path)}},
                               'protenix': {'model_name': model_name, 'use_constraints': False, 'load_checkpoint_path': str(tmp_path / 'weights.pt')}})
    result = construct_evaluators(config, raw_model_results_dir=tmp_path, oligo_names=['A'])
    assert captured[0][0]['model_name'] == model_name
    assert captured[0][1]['source_dir'] == str(tmp_path)
    assert result['p'].requested_model_name == model_name
    assert result['p'].effective_model_name == model_name
