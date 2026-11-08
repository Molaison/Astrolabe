"""CPU checks for data-only interface inputs and pinned native root calibration."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from branching_test_support import native_control_assets
from omegaconf import OmegaConf, open_dict

from astrolabe.interface import _prepare, _read_structure, _normalized_text, run_interface
from astrolabe.protein_beam import load_config
from astrolabe.protein_beam_calibration import calibrate_root, geometry_root
from astrolabe.protein_beam_geometry import _coordinates, pose_geometry

PROJECT = Path('/case')
INPUT = PROJECT / 'inputs/001_binder_interface/01_astrolabe_beam/03_bind_target_homologies'
PREVIOUS = PROJECT / 'config/001_binder_interface/01_astrolabe_beam/12_config_hierarchy'
TASK = Path(__file__).resolve().parents[1] / 'config/interface.yaml'
CASES = ['D5056_D15', 'D6542_D32', 'D6998_D19']


def test_directory_preflight_extracts_all_inputs_and_keeps_homology(tmp_path):
    plan, files = _prepare(INPUT, INPUT/'gd-ctd_data.json', tmp_path, None, None)
    assert [c['name'] for c in plan['cases']] == CASES
    for c in plan['cases']:
        assert (c['binder_chain'], c['target_chain']) == ('A', 'B')
        assert (len(c['binder_sequence']), len(c['target_sequence'])) == (73, 198)
        cfg = OmegaConf.create(files[c['config']])
        assert cfg.protein_beam.max_evaluations == 7
        assert cfg.protein_beam.root_calibration.enabled
        assert cfg.protein_beam.af3_call_budget.max_total == 7
        assert cfg.homologies.B == f'{INPUT}/gd-ctd_data.json:B'
        assert 'root_structure' not in cfg.case_input
        assert cfg.evaluator.alphafold3.num_recycles == 3
        assert cfg.mutator.mutator_terms.LigandMPNN_itf.temperature == .1
        assert cfg.mutator.mutator_terms.LigandMPNN_itf.ligand_mpnn_use_side_chain_context == 1


def test_prepare_resume_rejects_modified_artifact_without_compute(tmp_path, monkeypatch):
    import astrolabe.protein_beam as beam
    monkeypatch.setattr(beam, 'run_beam', lambda *a, **k: pytest.fail('Must not launch inference'))
    kwargs = dict(input_path=INPUT/'D5056_D15.pdb', target_homology=INPUT/'gd-ctd_data.json', output_root=tmp_path/'out', prepare_only=True)
    assert run_interface(**kwargs) == 0
    assert run_interface(**kwargs, resume=True) == 0
    with pytest.raises(FileExistsError): run_interface(**kwargs)
    manifest = tmp_path/'out/interface_plan.json'
    original_plan = manifest.read_text()
    changed_plan = json.loads(original_plan)
    changed_plan['implementation']['interface.py'] = 'later-source'
    manifest.write_text(json.dumps(changed_plan))
    with pytest.raises(ValueError, match='implementation.*changed'):
        run_interface(**kwargs, resume=True)
    manifest.write_text(original_plan)
    cfg = tmp_path/'out/prepared/D5056_D15/config.yaml'
    cfg.write_text(cfg.read_text() + '\n# altered\n')
    with pytest.raises(ValueError, match='changed or missing'): run_interface(**kwargs, resume=True)


def test_batch_invalid_input_fails_before_creating_output(tmp_path):
    data = tmp_path/'inputs'; data.mkdir()
    (data/'valid.pdb').write_bytes((INPUT/'D5056_D15.pdb').read_bytes())
    (data/'bad.pdb').write_text('END\n')
    with pytest.raises(ValueError):
        run_interface(input_path=data, target_homology=INPUT/'gd-ctd_data.json', output_root=tmp_path/'out', prepare_only=True)
    assert not (tmp_path/'out').exists()


def test_normalization_supports_renamed_chains_without_coordinate_change(tmp_path):
    source = (INPUT/'D5056_D15.pdb').read_text()
    changed = '\n'.join(line[:21] + {'A':'X','B':'Y'}.get(line[21], line[21]) + line[22:] if line.startswith(('ATOM  ','TER   ')) and len(line)>21 else line for line in source.splitlines()) + '\n'
    file = tmp_path/'renamed.pdb'; file.write_text(changed)
    plan, artifacts = _prepare(file, INPUT/'gd-ctd_data.json', tmp_path/'output', None, None)
    assert plan['cases'][0]['target_chain'] == 'Y'
    assert plan['cases'][0]['binder_chain'] == 'X'
    normalized = tmp_path/'normalized.cif'
    normalized.write_text(artifacts['prepared/renamed/input.cif'])
    original = _coordinates(str(INPUT/'D5056_D15.pdb'))
    new = _coordinates(str(normalized))
    for chain in ('A','B'): np.testing.assert_array_equal(original[chain], new[chain])


def test_ambiguous_target_and_duplicate_case_ids_fail_early(tmp_path):
    v=json.loads((INPUT/'gd-ctd_data.json').read_text())
    v['sequences'].append(copy.deepcopy(v['sequences'][0])); v['sequences'][-1]['protein']['id']='Z'
    homology=tmp_path/'homology.json'; homology.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='ambiguous'):
        _prepare(INPUT/'D5056_D15.pdb',homology,tmp_path/'out',None,None)
    assert len(_prepare(INPUT/'D5056_D15.pdb',str(homology)+':B',tmp_path/'out',None,None)[0]['cases'])==1
    data=tmp_path/'dupes'; data.mkdir()
    for extension in ('pdb','cif'): (data/f'case.{extension}').write_text('')
    with pytest.raises(ValueError,match='colliding'):
        _prepare(data,INPUT/'gd-ctd_data.json',tmp_path/'out',None,None)


@pytest.mark.parametrize('case', CASES)
def test_automatic_calibration_reproduces_accepted_limits(case,tmp_path):
    previous=load_config(PREVIOUS/f'03_{case}.yaml')
    record=json.loads(Path(previous.protein_beam.root_parity_reference).read_text())
    cfg=load_config(TASK)
    with open_dict(cfg):
        cfg.seqs=previous.seqs
        cfg.case_input.structure=previous.case_input.structure
    record['geometry']=pose_geometry(cfg.case_input.structure,record['structure_path'],record['structure_path'])
    state={'status':'running','generation':0,'proposals':0,'records':[record]}
    terminal=tmp_path/'candidates/000000/result.json'; terminal.parent.mkdir(parents=True)
    terminal.write_text(json.dumps(record))
    evidence=calibrate_root(cfg,tmp_path,state,record)
    expected=OmegaConf.to_container(previous.protein_beam.gates,resolve=True)
    for group in expected:
        assert evidence['gates'][group] == pytest.approx(expected[group], rel=0, abs=1e-12)
    assert geometry_root(cfg,tmp_path,{'id':1},'unused')==record['structure_path']
    assert calibrate_root(cfg,tmp_path,state,record)==evidence
    saved=json.loads((tmp_path/'root_calibration.json').read_text())
    saved['fingerprint']['atoms']='corrupted'
    (tmp_path/'root_calibration.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='changed'): calibrate_root(cfg,tmp_path,state,record)


@pytest.mark.usefixtures("native_control_assets")
def test_completed_root_resume_verifies_context_without_replay(tmp_path, monkeypatch):
    from concurrent.futures import Future
    import astrolabe.protein_beam as beam
    previous=load_config(PREVIOUS/'03_D5056_D15.yaml')
    root=json.loads(Path(previous.protein_beam.root_parity_reference).read_text())
    cfg=load_config(TASK)
    with open_dict(cfg):
        cfg.seqs=previous.seqs
        cfg.homologies=previous.homologies
        cfg.case_input.structure=previous.case_input.structure
        cfg.protein_beam.max_evaluations=1
        cfg.out_dir=str(tmp_path/'beam')
    root['geometry']=pose_geometry(cfg.case_input.structure,root['structure_path'],root['structure_path'])
    calls=[]
    class Pool:
        def __init__(self,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def submit(self,fn,task):
            import shutil
            calls.append(task['kind'])
            if task['kind'] == 'evaluate' and cfg.protein_beam.get('af3_call_budget'):
                from astrolabe.protein_beam_budget import claim_call
                claim_call(cfg.protein_beam.af3_call_budget, task)
            worker=tmp_path/'beam/workers/fixture'
            model=worker/'structures'/Path(root['structure_path']).name
            model.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root['structure_path'],model)
            label=model.name.removesuffix('_model.cif')
            old=Path(root['structure_path']).parent.parent/'raw_model_results/alphafold3'/label/(label+'_confidences.json')
            raw=worker/'raw_model_results/alphafold3'/label/old.name
            raw.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(old,raw)
            snapshot=tmp_path/'beam/candidates/000000/scores.pkl'
            snapshot.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(root['snapshot'],snapshot)
            value=dict(task,**{k:root[k] for k in ('sequence','seed','metrics','holes','geometry','score')},status='done',snapshot=str(snapshot),structure_path=str(model))
            from astrolabe.protein_beam_contract import native_artifact_manifest
            value['native_artifacts']=native_artifact_manifest(value)
            (snapshot.parent/'result.json').write_text(json.dumps(value))
            f=Future();f.set_result(value);return f
    monkeypatch.setattr(beam,'ProcessPoolExecutor',Pool)
    output=tmp_path/'beam'
    result=beam.run_beam(cfg,output,'unused',workers=2,in_flight=2)
    assert result['budget_complete'] and result['calibrated_gates']
    assert beam.run_beam(cfg,output,'unused',workers=2,in_flight=2,resume=True)['status']=='complete'
    assert calls==['evaluate']
    import astrolabe.protein_beam_contract as contract_module
    real_contract = contract_module.scientific_contract
    def changed_contract(value):
        contract, digest = real_contract(value)
        changed = copy.deepcopy(contract)
        changed['source_sha256']['interface.py'] = 'changed'
        changed_digest = hashlib.sha256(
            json.dumps(changed, sort_keys=True).encode()
        ).hexdigest()
        return changed, changed_digest
    monkeypatch.setattr(contract_module, 'scientific_contract', changed_contract)
    with pytest.raises(ValueError, match='implementation changed'):
        beam.run_beam(cfg, output, 'unused', workers=2, in_flight=2, resume=True)
    monkeypatch.setattr(contract_module, 'scientific_contract', real_contract)
    assert calls == ['evaluate']
    native = saved = json.loads((output/'beam_state.json').read_text())
    record = native['records'][0]
    for key in ('structure_path', 'snapshot'):
        path = Path(record[key])
        original_bytes = path.read_bytes()
        path.write_bytes(original_bytes + b'\ntampered\n')
        with pytest.raises(ValueError, match='content changed'):
            beam.run_beam(cfg, output, 'unused', workers=2, in_flight=2, resume=True)
        path.write_bytes(original_bytes)
    terminal = output/'candidates/000000/result.json'
    terminal_bytes = terminal.read_bytes()
    terminal.unlink()
    with pytest.raises(FileNotFoundError, match='terminal result is missing'):
        beam.run_beam(cfg, output, 'unused', workers=2, in_flight=2, resume=True)
    terminal.write_bytes(terminal_bytes)
    assert calls == ['evaluate']
    for filename in ('final_beam.json','scores.csv','pareto_archive.json'):
        (output/filename).unlink()
    beam.run_beam(cfg,output,'unused',workers=2,in_flight=2,resume=True)
    assert all((output/f).exists() for f in ('final_beam.json','scores.csv','pareto_archive.json'))
    checkpoint=output/'beam_state.json'; saved=json.loads(checkpoint.read_text())
    broken=copy.deepcopy(saved); broken['records']=[]
    checkpoint.write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='exactly one'):beam.run_beam(cfg,output,'unused',workers=2,in_flight=2,resume=True)
    checkpoint.write_text(json.dumps(saved))
    terminal=output/'candidates/000000/result.json'; original=terminal.read_text()
    changed=json.loads(original); changed['seed']+=1; terminal.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='dispatched seed'):beam.run_beam(cfg,output,'unused',workers=2,in_flight=2,resume=True)
    terminal.write_text(original)
    changed=json.loads(original); del changed['metrics']; terminal.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='required fields'):beam.run_beam(cfg,output,'unused',workers=2,in_flight=2,resume=True)
    terminal.write_text(original)
    calibration=output/'root_calibration.json'; frozen=calibration.read_text(); calibration.unlink()
    missing=copy.deepcopy(saved); missing.pop('calibrated_gates'); checkpoint.write_text(json.dumps(missing))
    with pytest.raises(ValueError,match='missing'):beam.run_beam(cfg,output,'unused',workers=2,in_flight=2,resume=True)
    assert not calibration.exists()
    calibration.write_text(frozen); checkpoint.write_text(json.dumps(saved))
    broken=copy.deepcopy(saved); broken['beam'][0]['snapshot']='corrupted-beam-copy'
    checkpoint.write_text(json.dumps(broken))
    repaired=beam.run_beam(cfg,output,'unused',workers=2,in_flight=2,resume=True)
    assert repaired['beam'][0]['snapshot']==repaired['records'][0]['snapshot']
    checkpoint.write_text(json.dumps(saved))
    broken=copy.deepcopy(saved); broken['status']='interrupted'
    parent=copy.deepcopy(broken['records'][0]); parent['snapshot']='wrong-snapshot'
    broken['generation_work']={'parents':[parent],'tasks':[]}
    checkpoint.write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='generation parent'):beam.run_beam(cfg,output,'unused',workers=2,in_flight=2,resume=True)
    checkpoint.write_text(json.dumps(saved))
    # Recover the CPU selection transition after root calibration was interrupted.
    recovering=copy.deepcopy(saved); recovering.update(status='interrupted',beam=[])
    checkpoint.write_text(json.dumps(recovering))
    recovered=beam.run_beam(cfg,output,'unused',workers=2,in_flight=2,resume=True)
    assert recovered['beam'] and recovered['beam'][0]['id']==0
    assert calls==['evaluate']
    calibration=output/'root_calibration.json'; calibration.unlink()
    with pytest.raises(ValueError,match='missing'):
        beam.run_beam(cfg,output,'unused',workers=2,in_flight=2,resume=True)
    assert calls==['evaluate']


def test_native_cli_dispatch_uses_data_flags(monkeypatch,tmp_path):
    import astrolabe.cli as cli
    import astrolabe.interface as interface
    seen={}
    monkeypatch.setattr(interface,'run_interface',lambda **kw: seen.update(kw) or 0)
    monkeypatch.setenv('GH_CUDA_PREINITIALIZE','1')
    import ctypes
    monkeypatch.setattr(ctypes,'CDLL',lambda *a, **kw: pytest.fail('CPU interface must not initialize CUDA'))
    with pytest.raises(SystemExit) as status:
        cli.main(['refine','interface','--input','structures','--target-homology','target.json','--out-dir',str(tmp_path),'--prepare-only'])
    assert status.value.code==0
    assert seen['input_path']=='structures' and seen['prepare_only']


def test_full_confidence_validation_uses_actual_residue_count(tmp_path,monkeypatch):
    import astrolabe.protein_beam_root_parity as parity
    structure=tmp_path/'structures/step_model.cif'
    raw=tmp_path/'raw_model_results/alphafold3/step/step_confidences.json'
    raw.parent.mkdir(parents=True)
    monkeypatch.setattr(parity,'_atoms',lambda p:[['A',1,'ALA','CA',0,0,0],['B',1,'ALA','CA',0,0,0]])
    raw.write_text(json.dumps({'pae':[[0,1],[1,0]]}))
    assert len(parity._confidence({'structure_path':str(structure)})['pae'])==2
    raw.write_text(json.dumps({'pae':[[0]]}))
    with pytest.raises(ValueError,match='2x2'):parity._confidence({'structure_path':str(structure)})


def test_directory_partial_resume_does_not_replay_completed_case(tmp_path,monkeypatch):
    import socket
    import tempfile
    import astrolabe.protein_beam as beam
    from astrolabe import protein_beam_recovery
    calls=[]; fail=[True]
    def fake(cfg, output, socket_path, **kwargs):
        output=Path(output); checkpoint=output/'beam_state.json'
        if checkpoint.exists():return json.loads(checkpoint.read_text())
        calls.append(cfg.task_name)
        if cfg.task_name.startswith('D6542') and fail[0]:
            fail[0]=False
            raise RuntimeError('mock interruption before inference')
        output.mkdir(parents=True,exist_ok=True)
        value=dict(status='complete',records=[{'id':0}],budget_complete=False,stop_reason='no_feasible_parent')
        checkpoint.write_text(json.dumps(value))
        return value
    monkeypatch.setattr(beam,'run_beam',fake)
    monkeypatch.setattr(
        protein_beam_recovery,
        'validate_completed_beam',
        lambda cfg, output, state, **kwargs: state,
    )
    with tempfile.TemporaryDirectory(prefix='interface-sock-') as rpc:
        owner=socket.socket(socket.AF_UNIX)
        path=Path(rpc)/'owner.sock'; owner.bind(str(path))
        args=dict(input_path=INPUT,target_homology=INPUT/'gd-ctd_data.json',output_root=tmp_path/'out',socket_path=str(path))
        try:
            with pytest.raises(RuntimeError,match='mock interruption'):run_interface(**args)
            assert calls==['D5056_D15_interface','D6542_D32_interface']
            assert run_interface(**args,resume=True)==0
            assert calls==['D5056_D15_interface','D6542_D32_interface','D6542_D32_interface','D6998_D19_interface']
        finally:owner.close()
    # Completed outputs require no running GPU service, but still verify plans.
    assert run_interface(**args,resume=True)==0
    assert len(calls)==4
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (tmp_path/'out').rglob('*') if p.is_file()}
    no_socket={key:value for key,value in args.items() if key!='socket_path'}
    from astrolabe.evaluators import af3_lifecycle
    monkeypatch.setattr(beam,'run_beam',lambda *a,**k:pytest.fail('Completed resume entered beam execution'))
    monkeypatch.setattr(af3_lifecycle,'recover_owned_services',
                        lambda *a,**k:pytest.fail('Completed resume inspected a service'))
    monkeypatch.setattr(af3_lifecycle,'OwnedAF3',
                        lambda *a,**k:pytest.fail('Completed resume created an AF3 owner'))
    assert run_interface(**no_socket,resume=True)==0
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in (tmp_path/'out').rglob('*') if p.is_file()}
    assert after==before


def test_status_only_complete_output_cannot_bypass_completed_preflight(tmp_path, monkeypatch):
    from astrolabe.evaluators import af3_lifecycle
    import astrolabe.protein_beam as beam

    args = dict(
        input_path=INPUT/'D5056_D15.pdb',
        target_homology=INPUT/'gd-ctd_data.json',
        output_root=tmp_path/'out',
        prepare_only=True,
    )
    assert run_interface(**args) == 0
    case = tmp_path/'out/cases/D5056_D15'
    case.mkdir(parents=True)
    (case/'beam_state.json').write_text(json.dumps({
        'status': 'complete',
        'records': [{'id': 0, 'status': 'done'}],
        'pending': [],
    }))
    monkeypatch.setattr(
        beam,
        'run_beam',
        lambda *args, **kwargs: pytest.fail('Forged complete output reached beam execution'),
    )
    monkeypatch.setattr(
        af3_lifecycle,
        'OwnedAF3',
        lambda *args, **kwargs: pytest.fail('Forged complete output created an AF3 owner'),
    )
    with pytest.raises(ValueError):
        run_interface(
            **{key: value for key, value in args.items() if key != 'prepare_only'},
            resume=True,
        )
