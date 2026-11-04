"""Audit completeness without loading models or substituting CPU certification."""
from __future__ import annotations

import importlib.util
import hashlib
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
    assert matrix['registry_status_counts']=={'DONE':0,'PARTIAL':6,'BLOCKED':28}
    control=matrix['runner_controls'][0]
    assert control['api']=='build_runner("ligandmpnn", model_type="protein_mpnn")'
    assert control['registry_certification'] is False
    assert control['planned_gpu_control']['status']=='NOT_RUN'
    assert control['planned_gpu_control']['evidence'] is None
    assert control['planned_gpu_control']['design_requests']==30
    assert control['planned_gpu_control']['trials']==10
    assert control['cpu_preflight']['native_sources_match_checkout']
    assert control['cpu_preflight']['state_tensor_count']==118
    assert control['cpu_preflight']['parameter_count']==1660485
    assert control['cpu_preflight']['native_parse']['mask_sum']==354
    assert control['cpu_preflight']['model_forward_calls']==0
    assert control['cpu_preflight']['gpu_execution'] is False
    assert rows['mutator','proteinmpnn']['status']=='BLOCKED'
    assert rows['mutator','ligandmpnn']['status']=='BLOCKED'
    assets=matrix['f102_asset_preflight']
    assert assets['esmfold']['main_checkpoint']['bytes']==2097152000
    assert assets['esmfold']['main_checkpoint']['zipfile_is_zipfile'] is False
    assert assets['esmfold']['actual_esm_or_openfold_import_executed'] is False
    assert assets['esmfold']['installation_metadata']['versions']['fair-esm'] is None


@pytest.mark.parametrize('changes', [
    {'status':'running'}, {'exitCode':1}, {'outputTruncated':True},
])
def test_incomplete_or_failed_preflight_cannot_be_consumed_as_success(changes):
    result={'status':'exited','exitCode':0,'outputTruncated':False,'stdout':'{"observed":true}'}
    result.update(changes)
    with pytest.raises(ValueError,match='incomplete, failed or truncated'):
        certification._successful_remote_json(result)


def test_cpu_summary_must_match_completed_raw_command_even_with_updated_manifest(tmp_path):
    relative=Path('evidence/runtime-next/agent-b/f102-backend-preflight')
    directory=tmp_path/relative
    shutil.copytree(ROOT/relative,directory)
    summary=directory/'CPU_COMPATIBILITY.json'
    value=json.loads(summary.read_text())
    value['checkpoint_and_structure']['model_forward_calls']=1
    summary.write_text(json.dumps(value))
    manifest=directory/'manifest.json'
    identities=json.loads(manifest.read_text())
    identities['files'][summary.name]={'bytes':summary.stat().st_size,'sha256':hashlib.sha256(summary.read_bytes()).hexdigest()}
    manifest.write_text(json.dumps(identities))
    with pytest.raises(ValueError,match='summary differs from its actual completed command output'):
        certification.f102_preflight(tmp_path)


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
