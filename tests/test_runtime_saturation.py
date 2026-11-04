"""Independent acceptance via actual Unix sockets and real CPU daemons.

ASTROLABE_EXPECT_GUARDRAILS=1 turns unavailable repair APIs into failures;
without it the frozen baseline runs characterization and explicitly skips
acceptance cases whose public contract is not implemented yet.
"""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


FIXTURE = Path(__file__).with_name('runtime_saturation_fixture.py')


def contract(module, attribute):
    try:
        value = getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError):
        message = 'repair contract not available: ' + module + '.' + attribute
        if os.environ.get('ASTROLABE_EXPECT_GUARDRAILS') == '1':
            pytest.fail(message)
        pytest.skip(message)
    return value


def run_scenario(name, tmp_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void',
               HIP_VISIBLE_DEVICES='', ROCR_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
               GH_CUDA_PREINITIALIZE='0', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for key in ('ASTROLABE_EXECUTION_ENDPOINT', 'ASTROLABE_EXECUTION_TOKEN', 'ASTROLABE_CPU_OWNER'):
        env.pop(key, None)
    # Each test owns an isolated scenario and enforces a parent deadline too.
    result = subprocess.run([sys.executable, str(FIXTURE), '--scenario', name, '--out', str(tmp_path)],
                            env=env, capture_output=True, text=True, timeout=42)
    (tmp_path / 'scenario.stdout').write_text(result.stdout)
    (tmp_path / 'scenario.stderr').write_text(result.stderr)
    assert result.returncode == 0, result.stderr[-10000:]
    report = json.loads((tmp_path / 'result.json').read_text())
    assert report['success'], report
    return report


def test_real_daemon_fixture_cpu_smoke(tmp_path):
    run_scenario('smoke', tmp_path)


def test_ready_candidates_two_evaluators_progress_while_producer_waits(tmp_path):
    from astrolabe.runtime.admission import available_cpus
    if available_cpus() < 2:
        pytest.skip('independent overlap requires >=2 allocated CPUs')
    report = run_scenario('ready_consumers', tmp_path)
    assert [item['completed'] for item in report['result']['evaluators']] == [4, 4]


def test_32_socket_business_requests_allow_nested_control_and_release(tmp_path):
    contract('astrolabe.runtime.broker_transport', 'endpoint_for')
    contract('astrolabe.runtime.client', 'AdmissionRejected')
    assert callable(contract('astrolabe.runtime.broker_client', 'BrokerClient').cancel)
    run_scenario('saturation', tmp_path)


def test_control_socket_lane_rejects_at_its_own_bound(tmp_path):
    contract('astrolabe.runtime.broker_transport', 'endpoint_for')
    run_scenario('bounded_control', tmp_path)


def test_pending_cancel_preserves_active_active_cancel_only_terminates_target(tmp_path):
    client = contract('astrolabe.runtime.broker_client', 'BrokerClient')
    if not hasattr(client, 'cancel'):
        if os.environ.get('ASTROLABE_EXPECT_GUARDRAILS') == '1':
            pytest.fail('repair contract not available: BrokerClient.cancel')
        pytest.skip('repair contract not available: BrokerClient.cancel')
    run_scenario('cancel_isolation', tmp_path)


def test_deterministic_application_error_preserves_other_cpu_worker(tmp_path):
    contract('astrolabe.runtime.client', 'ApplicationError')
    run_scenario('application_error', tmp_path)
