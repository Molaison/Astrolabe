"""Configuration composition must preserve the accepted scientific settings."""
from copy import deepcopy
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest

import astrolabe.protein_beam as beam


ROOT = Path('/case/config/001_binder_interface/01_astrolabe_beam')


@pytest.mark.parametrize('case', ['D6998_D19', 'D6542_D32', 'D5056_D15'])
def test_inherited_case_preserves_accepted_science(case):
    old = beam.load_config(ROOT/'11_mpnn_pose_comparison'/f'14_{case}_ligandmpnn.yaml')
    new = beam.load_config(ROOT/'12_config_hierarchy'/f'03_{case}.yaml')
    beam.validate_config(old)
    beam.validate_config(new)
    left, right = (OmegaConf.to_container(cfg, resolve=False) for cfg in (old, new))
    # The migrated recipe declares its route; scientific inputs must still
    # match the accepted standalone recipe exactly.
    assert left.get('search') is None
    assert right['search']['algorithm'] == 'branching'
    assert right['search']['owner'] == 'isolated_native'
    right.pop('search')
    # Output binding and resource defaults used to live in the launch command.
    for value in (left, right):
        value.pop('out_dir')
        value.pop('raw_model_results_dir')
        value['protein_beam'].pop('workers')
        value['protein_beam'].pop('in_flight')
        value['protein_beam']['af3_call_budget'].pop('ledger')
        value['protein_beam']['af3_call_budget'].pop('scope')
    assert left == right
    assert new.protein_beam.workers == new.protein_beam.in_flight == 2


def test_case_overrides_task_without_changing_base_defaults(tmp_path):
    case = tmp_path/'override.yaml'
    case.write_text('''defaults: [binder_interface_beam, _self_]
mutator:
  mutator_terms:
    LigandMPNN_itf:
      temperature: 0.2
protein_beam:
  width: 5
  workers: 4
''')
    cfg = beam.load_config(case)
    assert cfg.mutator.mutator_terms.LigandMPNN_itf.temperature == .2
    assert cfg.protein_beam.width == 5
    assert cfg.protein_beam.workers == 4
    assert cfg.protein_beam.in_flight == 2
    base = tmp_path/'base_case.yaml'
    base.write_text('defaults: [base, _self_]\n')
    cfg = beam.load_config(base)
    assert cfg.protein_beam.workers == cfg.protein_beam.in_flight == 3
    assert cfg.mutator.LigandMPNN.temperature == .3
    assert cfg.evaluator.alphafold3.num_recycles == 10
    assert cfg.evaluator.alphafold3.num_samples == 5


def test_owner_and_client_read_same_task_defaults():
    from astrolabe.evaluators.af3_service import load_service_config
    path = ROOT/'12_config_hierarchy/03_D5056_D15.yaml'
    client = beam.load_config(path).evaluator.alphafold3
    owner = load_service_config(path)
    assert OmegaConf.to_container(owner) == OmegaConf.to_container(client, resolve=True)
    legacy = load_service_config(ROOT/'11_mpnn_pose_comparison/03_af3_owner.yaml')
    assert legacy.num_recycles == owner.num_recycles == 3
    assert legacy.num_samples == owner.num_samples == 1
    assert legacy.num_seeds == owner.num_seeds == 1


def test_worker_claims_the_coordinator_ledger_after_output_relocation(tmp_path):
    from astrolabe.protein_beam_budget import reserve_calls, claim_call
    cfg = beam.load_config(ROOT/'12_config_hierarchy/03_D5056_D15.yaml')
    cfg.out_dir = str(tmp_path/'run')
    task = dict(kind='evaluate', id=0, seed=10079)
    admitted, refused = reserve_calls(cfg.protein_beam.af3_call_budget, [task])
    assert not refused and len(admitted) == 1
    worker = beam.load_worker_config(OmegaConf.to_yaml(cfg, resolve=False))
    worker.out_dir = str(tmp_path/'run/workers/worker1')
    claim_call(worker.protein_beam.af3_call_budget, admitted[0])
    import json
    ledger = json.loads((tmp_path/'run/af3_calls.json').read_text())
    assert ledger['entries'][admitted[0]['af3_call_token']]['status'] == 'issued'
    assert not (tmp_path/'run/workers/worker1/af3_calls.json').exists()


def test_owner_clears_only_the_full_task_client_socket(tmp_path):
    from astrolabe.evaluators.af3_service import load_service_config, load_runner
    path = tmp_path/'case.yaml'
    path.write_text('''defaults: [binder_interface_beam, _self_]
evaluator:
  alphafold3:
    service_socket: /tmp/client-owner.sock
''')
    assert beam.load_config(path).evaluator.alphafold3.service_socket == '/tmp/client-owner.sock'
    assert load_service_config(path).service_socket is None
    raw = tmp_path/'raw.yaml'
    raw.write_text('service_socket: /tmp/client-owner.sock\n')
    legacy = load_service_config(raw)
    assert legacy.service_socket == '/tmp/client-owner.sock'
    with pytest.raises(ValueError, match='Owner config'):
        load_runner(legacy)


def test_native_owner_cli_loads_inherited_config_without_an_extra_owner_yaml(monkeypatch):
    import astrolabe.evaluators.af3_service as service
    calls = []
    runner = object()
    def load(config):
        calls.append(('load', config.num_recycles, config.num_samples))
        return runner
    handler = object()
    monkeypatch.setattr(service, 'load_runner', load)
    monkeypatch.setattr(service, 'inference_handler', lambda value, config: handler)
    monkeypatch.setattr(service, 'serve', lambda value, socket, pending: calls.append(('serve', value, socket, pending)))
    service.main(['--config', str(ROOT/'12_config_hierarchy/03_D5056_D15.yaml'), '--socket', '/tmp/owned.sock'])
    assert calls == [('load', 3, 1), ('serve', handler, '/tmp/owned.sock', 4)]


@pytest.mark.parametrize('overrides,expected', [([], (2, 2)), (['--workers', '4', '--in-flight', '3'], (4, 3))])
def test_existing_cli_uses_config_defaults_and_explicit_overrides(monkeypatch, tmp_path, overrides, expected):
    cfg = beam.load_config(ROOT/'12_config_hierarchy/03_D5056_D15.yaml')
    cfg.evaluator.alphafold3.service_socket = '/tmp/configured-owner.sock'
    monkeypatch.setattr(beam, 'load_config', lambda path: deepcopy(cfg))
    calls = []
    def run(value, output, socket, **kwargs):
        calls.append((value, output, socket, kwargs))
        return dict(status='complete', records=[], budget_complete=True, wall_s=0.)
    monkeypatch.setattr(beam, 'run_beam', run)
    monkeypatch.setattr(sys, 'argv', ['protein_beam', '--config', 'unused.yaml', '--out', str(tmp_path), *overrides])
    assert beam.main() == 0
    value, output, socket, kwargs = calls[0]
    assert (kwargs['workers'], kwargs['in_flight']) == expected
    assert socket == '/tmp/configured-owner.sock'
    assert value.protein_beam.af3_call_budget.ledger == str(tmp_path/'af3_calls.json')
    assert value.protein_beam.af3_call_budget.scope == value.task_name


def test_socket_cli_override_and_missing_owner_fail_before_compute(monkeypatch, tmp_path):
    cfg = beam.load_config(ROOT/'12_config_hierarchy/03_D5056_D15.yaml')
    monkeypatch.setattr(beam, 'load_config', lambda path: deepcopy(cfg))
    def run(*args, **kwargs):
        assert args[2] == '/tmp/explicit-owner.sock'
        return dict(status='complete', records=[], budget_complete=True, wall_s=0.)
    monkeypatch.setattr(beam, 'run_beam', run)
    argv = ['protein_beam', '--config', 'unused.yaml', '--out', str(tmp_path)]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit) as exc:
        beam.main()
    assert exc.value.code == 2
    monkeypatch.setattr(sys, 'argv', argv + ['--service-socket', '/tmp/explicit-owner.sock'])
    assert beam.main() == 0


def test_cli_expands_output_once_for_results_and_ledger(monkeypatch):
    cfg = beam.load_config(ROOT/'12_config_hierarchy/03_D5056_D15.yaml')
    monkeypatch.setattr(beam, 'load_config', lambda path: deepcopy(cfg))
    expected = str(Path('~/binder-config-check').expanduser().resolve())
    def run(value, output, socket, **kwargs):
        assert value.out_dir == output == expected
        assert value.protein_beam.af3_call_budget.ledger == expected+'/af3_calls.json'
        return dict(status='complete', records=[], budget_complete=True, wall_s=0.)
    monkeypatch.setattr(beam, 'run_beam', run)
    monkeypatch.setattr(sys, 'argv', ['protein_beam', '--config', 'unused.yaml', '--out', '~/binder-config-check', '--service-socket', '/tmp/owner.sock'])
    assert beam.main() == 0
