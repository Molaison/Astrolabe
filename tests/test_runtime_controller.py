"""Control decisions on explicit observed traces, not throughput benchmarks."""
from astrolabe.runtime.controller import AdaptiveController
import pytest


def observations(times, *, cold=False, shape='128', ready=0, failures=0, pressure=0):
    return {'recent': [{'time': t, 'backend': 'exact-model-a', 'shape_key': shape,
                         'cold_start': cold, 'phase': 'device', 'outcome': 'completed'} for t in times],
            'waiting': ready, 'failures': failures,
            'ledger': {'devices': {}, 'limits': {'ram_bytes': 1000, 'cpu_slots': 2}},
            'observation': {'devices': {}, 'rss_bytes': int(pressure * 1000)}}


def control():
    return AdaptiveController(4, started=0, policy={'window_seconds': 1, 'min_completions': 2,
                                                  'cooldown_seconds': 1})


def test_independent_backlog_can_fill_starved_gpu_without_extra_trial_work():
    controller = control()
    assert controller.observe(observations([.2, .8]), now=1, independent_work=True) == 2
    assert controller.last_reason == 'independent_supply_trial'
    assert controller.observe(observations([1.2, 1.3, 1.6, 1.9]), now=2) == 2
    assert controller.last_reason == 'throughput_gain_retained'


def test_busy_or_cold_alone_cannot_increase_concurrency():
    controller = control()
    assert controller.observe(observations([.2, .8], cold=True), now=1) == 1
    assert controller.observe(observations([1.2, 1.8], ready=2), now=2) == 1
    assert controller.last_reason == 'device_supplied'


def test_trial_regression_and_oom_retreat_without_replaying_work():
    controller = control()
    controller.observe(observations([.2, .8]), now=1)
    assert controller.observe(observations([1.2, 2.9]), now=3) == 1
    assert controller.last_reason == 'throughput_regression'
    assert controller.observe(observations([3.2, 3.8]), now=3.9) == 1
    controller.limit = 4
    assert controller.observe(observations([], failures=1), now=4) == 2
    assert controller.last_reason == 'execution_failure'


def test_shape_change_does_not_become_false_concurrency_gain():
    controller = control()
    controller.observe(observations([.2, .8]), now=1)
    assert controller.observe(observations([1.1, 1.2, 1.3, 1.4], shape='64'), now=2) == 1
    assert controller.last_reason == 'workload_changed_during_trial'


def test_no_independent_supply_is_causal_idle_and_memory_pressure_shrinks():
    controller = control()
    assert controller.observe(observations([.2, .8]), now=1, independent_work=False) == 1
    assert controller.last_reason == 'causal_dependency_or_no_independent_supply'
    controller.limit = 4
    assert controller.observe(observations([], pressure=.95), now=2) == 2
    assert controller.last_reason == 'memory_pressure'


def test_long_cold_start_does_not_dilute_the_warm_throughput_baseline():
    controller = control()
    metrics = observations([9.2, 9.9])
    metrics['recent'].insert(0, observations([9.0], cold=True)['recent'][0])
    assert controller.observe(metrics, now=10.0) == 1
    assert controller.last_reason == 'cold_start_observed'
    assert controller.window_start == 9.0
    assert controller.observe(metrics, now=10.1) == 2
    # The denominator starts after cold execution; using ten seconds here
    # would manufacture a gain for the next steady window.
    assert controller.experiment['throughput'] == pytest.approx(2 / 1.1)


def test_new_worker_cold_start_resets_the_active_trial_observation_window():
    controller = control()
    controller.observe(observations([.2, .8]), now=1)
    trial = observations([1.3, 1.9])
    trial['recent'].insert(1, observations([1.8], cold=True)['recent'][0])
    assert controller.observe(trial, now=2) == 2
    assert controller.last_reason == 'cold_start_observed'
    assert [event['time'] for event in controller.window] == [1.9]
    assert controller.experiment is not None
