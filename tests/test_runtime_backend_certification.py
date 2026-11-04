"""Audit completeness without loading models or substituting CPU certification."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/validation/runtime_backend_certification.py'
spec = importlib.util.spec_from_file_location('backend_certification', SCRIPT)
certification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(certification)


def test_inventory_matches_actual_builtin_registry_imports():
    code = '''import json
import astrolabe.evaluators.utils
import astrolabe.mutators.base
from astrolabe.evaluators.registry import EvaluatorRegistry
from astrolabe.mutators.registry import MutatorRegistry
print(json.dumps({'evaluator':EvaluatorRegistry.list_available(),'mutator':MutatorRegistry.list_available()}))
'''
    imported = subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,
                              capture_output=True,check=True)
    actual = json.loads(imported.stdout)
    assert {kind:sorted(certification.discover(ROOT, kind)) for kind in actual} == actual
    assert (len(actual['evaluator']),len(actual['mutator'])) == (14,20)


def test_generator_runs_without_site_packages_and_preserves_evidence_scopes(tmp_path):
    output = tmp_path/'matrix.json'
    # -S excludes site packages: inventory must not require or initialize a
    # torch/JAX/model environment merely to report an unverified backend.
    subprocess.run([sys.executable,'-S',str(SCRIPT),'--output',str(output)],
                   cwd=ROOT,text=True,capture_output=True,check=True)
    matrix = json.loads(output.read_text())
    rows = {(row['kind'],row['key']):row for row in matrix['records']}
    assert len(rows) == 34
    assert all(not row['pretrained_gpu']['measured'] for row in rows.values())
    dummy = rows['evaluator','dummy']['historical_evidence']
    assert dummy['kind']=='historical_actual_cpu_workflow'
    assert dummy['completed_trajectories']==27
    esm = rows['evaluator','esmfold']
    assert esm['local_output_evidence']['kind']=='synthetic_torch_cpu_with_source_extracted_native_helpers'
    assert esm['pretrained_gpu']['status']=='BLOCKED'
    assert not esm['performance_certification']['effective_throughput_improvement_proved']
    assert esm['actual_cpu_process_evidence']['kind']=='real_cpu_processes_with_fake_model_and_synthetic_local_checkpoint'
    assert esm['actual_cpu_process_evidence']['test_cases']==65
    assert esm['actual_cpu_process_evidence']['torch']['cuda_build'] is None
    assert rows['mutator','mcmc']['recorded_w0_ci_evidence']['accepted_mutations']==0
    assert rows['mutator','mcmc']['recorded_w0_ci_evidence']['initial_evaluations']==90


@pytest.mark.parametrize('change', ['new_key','changed_factory'])
def test_registry_drift_cannot_silently_reuse_certification_scope(tmp_path, change):
    for relative, _ in certification.REGISTRIES.values():
        destination = tmp_path/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,destination)
    source = tmp_path/'astrolabe/evaluators/utils.py'
    value = source.read_text()
    if change=='new_key':
        value += '\n@register_evaluator_factory("unreviewed_backend")\ndef unreviewed_factory():\n    pass\n'
        expected='registry/catalog drift'
    else:
        # Select the first alphabetic entry so it fails before checking
        # unrelated evidence/test paths in this minimal temporary checkout.
        value = value.replace('def _build_alphafold2_evaluator(', 'def different_af2_factory(')
        expected='factory changed'
    source.write_text(value)
    with pytest.raises(ValueError,match=expected):
        certification.build_matrix(tmp_path)
