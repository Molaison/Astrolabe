"""Configuration reaches the real repeat lifecycle without manual worker tuning."""
import pytest
from omegaconf import OmegaConf
import run
from astrolabe.runtime import admission, session


def config(**execution):
    return OmegaConf.create({'task_name': 'auto', 'repeats': 4, 'seed': 17,
                             'execution': {'enabled': True, 'max_active': 'auto',
                                           'max_resident_models': 'auto', **execution}})


def test_auto_bounds_reach_runtime_with_original_repeat_seeds(monkeypatch):
    monkeypatch.setattr(run, 'install_handlers', lambda: None)
    monkeypatch.setattr(run, 'stop_requested', lambda: False)
    monkeypatch.setattr(admission, 'available_cpus', lambda: 2)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,5')
    monkeypatch.setattr(session, 'validate_parallel', lambda _: None)
    seen = {}
    def execute(configs, max_active, stop_requested, on_completed, **kwargs):
        seen.update(max_active=max_active, **kwargs)
        seen['seeds'] = [cfg.seed for cfg in configs]
        return 4
    monkeypatch.setattr(session, 'run_parallel', execute)
    run.run_repeats(config())
    assert seen == {'max_active': 2, 'max_resident_models': 2,
                    'max_resident_per_device': 1, 'seeds': [17, 18, 19, 20]}


def test_disabled_auto_never_probes_device_or_uses_parallel_runtime(monkeypatch):
    monkeypatch.setattr(run, 'install_handlers', lambda: None)
    monkeypatch.setattr(run, 'stop_requested', lambda: False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'invalid,allocation,unused')
    def forbidden(*args, **kwargs):
        pytest.fail('disabled execution must not resolve resource limits')
    monkeypatch.setattr(admission, 'resolve_active_limit', forbidden)
    monkeypatch.setattr(admission, 'resolve_resident_limit', forbidden)
    seen = []
    monkeypatch.setattr(run, '_run_pipeline', lambda cfg: seen.append(cfg.seed))
    run.run_repeats(config(enabled=False))
    assert seen == [17, 18, 19, 20]


@pytest.mark.parametrize('overrides', [{'max_active': False}, {'max_active': 0},
    {'max_resident_models': 'AUTO'}, {'max_resident_per_device': True}, {'enabled': 'false'}])
def test_execution_limits_reject_invalid_values(monkeypatch, overrides):
    monkeypatch.setattr(run, 'install_handlers', lambda: None)
    with pytest.raises(ValueError):
        run.run_repeats(config(**overrides))
