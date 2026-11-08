import hashlib
import json
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock
import sys
import pytest
import astrolabe.cli as cli
from astrolabe.posthoc.proton_states import run_from_config


def _inputs(tmp_path):
    source = tmp_path/'structure.pdb'
    source.write_text('retained input identity\n')
    manifest = tmp_path/'inputs.json'
    record = dict(item_id='parent', pdb_path=str(source),
                  input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  histidine_positions=[])
    manifest.write_text(json.dumps(dict(schema='astrolabe_proton_state_audit_v1', items=[record])))
    config = tmp_path/'config.yaml'
    config.write_text('mutator: {}\n')
    return config, manifest, record


def test_cli_dispatch_is_lazy_and_cpu(monkeypatch):
    module = ModuleType('astrolabe.posthoc.proton_states')
    module.run_from_config = Mock(return_value={'items': [], 'state_calls': 0})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ns = cli._build_parser().parse_args(['posthoc', 'proton-states', '--config', 'c.yaml',
        '--inputs', 'i.json', '--out-dir', 'new result', '--dry-run'])
    assert ns._cuda_preinitialize is False
    assert ns._handler(ns) == 0
    module.run_from_config.assert_called_once_with(config_path=Path('c.yaml'),
        inputs_path=Path('i.json'), output_dir=Path('new result'), dry_run=True)


@pytest.mark.parametrize("source_root", [None, "original source tree"])
def test_parity_dispatches_exact_native_inputs_without_cuda(monkeypatch, source_root):
    module = ModuleType('astrolabe.posthoc.proton_parity')
    module.main = Mock(return_value=0)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    args = ['--repo-root', 'official', '--checkpoint', 'weights', '--hbplus', 'hbplus',
            '--arm', 'clean', '--inputs', 'inputs.json', '--output', 'out']
    if source_root is not None:
        args.extend(["--source-root", source_root])
    ns = cli._build_parser().parse_args(['posthoc', 'proton-parity', *args])
    assert ns._cuda_preinitialize is False
    assert ns._handler(ns) == 0
    module.main.assert_called_once_with(args)


@pytest.mark.parametrize('dry_run', [True, False])
def test_zero_his_control_does_not_load_model_or_invent_gap(tmp_path, monkeypatch, dry_run):
    config, manifest, _ = _inputs(tmp_path)
    import astrolabe.mutators.base as base
    factory = Mock(side_effect=AssertionError('zero His must not load a model'))
    monkeypatch.setattr(base, '_build_proton_pottsmpnn_mutator', factory)
    output = tmp_path/'out'
    result = run_from_config(config_path=config, inputs_path=manifest, output_dir=output, dry_run=dry_run)
    factory.assert_not_called()
    assert result['state_calls'] == result['AF3_calls'] == 0
    if dry_run:
        assert not output.exists()
    else:
        assert result['items'][0]['result'] is None
        assert 'gap' not in result['items'][0]


def test_identity_failure_precedes_output_and_model(tmp_path):
    config, manifest, record = _inputs(tmp_path)
    Path(record['pdb_path']).write_text('different input\n')
    output = tmp_path/'out'
    with pytest.raises(ValueError, match='identity/hash mismatch'):
        run_from_config(config_path=config, inputs_path=manifest, output_dir=output)
    assert not output.exists()


def test_repeated_ids_are_rejected(tmp_path):
    config, manifest, record = _inputs(tmp_path)
    manifest.write_text(json.dumps(dict(schema='astrolabe_proton_state_audit_v1', items=[record, record])))
    with pytest.raises(ValueError, match='unique'):
        run_from_config(config_path=config, inputs_path=manifest, output_dir=tmp_path/'out')
