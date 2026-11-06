from copy import deepcopy
import json
from types import SimpleNamespace as NS
import numpy as np
import pytest
from astrolabe.protein_beam_policy import pareto_select, sequence_rejections, candidate_rejections
from branching_test_support import (
    native_branch_config,
    native_control_assets,
    prepare_one_generation,
    run_one_generation,
)


pytestmark = pytest.mark.usefixtures("native_control_assets")


def row(i, values):
    return dict(id=i,sequence=f'protein{i}',status='done',score=0.,
                metrics=dict(zip(('interface_ipae','interface_ipsae','interface_sc','interface_holes','interface_buns'),values)))


def test_pareto_avoids_weighted_holes_dominance_and_preserves_directions():
    weak=row(0,(5.,.8,.6,-100.,5.))
    dominant=row(1,(4.,.9,.7,-101.,4.))
    tradeoff=row(2,(3.,.85,.65,-90.,3.))
    result=pareto_select([weak,dominant,tradeoff],3)
    assert {c['id'] for c in result[:2]}=={1,2}
    assert result[-1]['id']==0 and result[-1]['selection']['pareto_front']==1
    dominant['eligible']=False
    assert 1 not in [c['id'] for c in pareto_select([weak,dominant],3)]


def test_pareto_crowding_is_stable_under_metric_unit_changes():
    candidates=[row(i,(float(i+1),.7+i/100,.6+i/50,-100.-i*10,8.-i)) for i in range(8)]
    expected=[c['id'] for c in pareto_select(deepcopy(candidates),4)]
    scaled=deepcopy(candidates)
    for c in scaled:
        c['metrics']['interface_holes']=c['metrics']['interface_holes']*1e6+12.
    assert [c['id'] for c in pareto_select(scaled,4)]==expected
    assert [c['id'] for c in pareto_select(list(reversed(scaled)),4)]==expected


def test_pareto_identical_metrics_dedup_and_invalid_input():
    a=row(0,(4.,.9,.7,-200.,0.));b=deepcopy(a);b.update(id=1,sequence='other')
    duplicate=deepcopy(a);duplicate['id']=3
    assert [c['id'] for c in pareto_select([b,duplicate,a],4)]==[0,1]
    a['metrics']['interface_holes']=float('nan')
    with pytest.raises(ValueError):pareto_select([a],1)


def test_forbidden_substitution_and_cumulative_limit_do_not_reject_existing_cys():
    assert sequence_rejections('CADA','CACA','CACA',['C'],2)==[]
    assert sequence_rejections('CACA','DADA','CACA',['C'],2)==[]
    assert sequence_rejections('CCCA','CACA','CACA',['C'],2)==['excluded_amino_acid']
    assert sequence_rejections('DDDD','CACA','CACA',['C'],2)==['total_mutation_limit']
    with pytest.raises(ValueError):sequence_rejections('CA','CACA','CACA',[],2)


def test_explicit_pose_and_confidence_limits():
    root=dict(interface_ipae=4.,interface_ipsae=.9)
    c=dict(geometry=dict(binder_pose_ca_rmsd=2.1),metrics=dict(interface_ipae=6.,interface_ipsae=.7))
    gates=dict(geometry_max=dict(binder_pose_ca_rmsd=2.),root_metric_max_ratio=dict(interface_ipae=1.25),
               root_metric_min_ratio=dict(interface_ipsae=.95))
    assert candidate_rejections(c,root,gates)==['binder_pose_ca_rmsd','interface_ipae','interface_ipsae']
    c['geometry']['binder_pose_ca_rmsd']=float('nan')
    with pytest.raises(ValueError):candidate_rejections(c,root,gates)


def test_native_mean_can_reverse_raw_sum_order_and_crowding_is_finite():
    a=row(0,(4.,.9,.7,-200.,2.));b=row(1,(4.,.9,.7,-150.,2.))
    a['holes']=dict(mean=-1.,atom_count=200)
    b['holes']=dict(mean=-1.5,atom_count=100)
    assert pareto_select(deepcopy([a,b]),1)[0]['id']==0
    assert pareto_select(deepcopy([a,b]),1,holes_ranking='per_atom')[0]['id']==1
    assert all(np.isfinite(c['selection']['crowding_distance']) for c in pareto_select([a,b],2,'per_atom'))


def _write_ca(path,a,b):
    lines=[];serial=0
    for chain,coordinates in [('A',a),('B',b)]:
        for index,point in enumerate(coordinates,1):
            serial+=1
            lines.append(f'ATOM  {serial:5d}  CA  ALA {chain}{index:4d}    {point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}  1.00 20.00           C')
    path.write_text('\n'.join(lines+['END','']))


def test_target_aligned_pose_detects_binder_motion_and_removes_global_motion(tmp_path):
    from astrolabe.protein_beam_geometry import pose_geometry
    b=np.array([[0.,0.,0.],[2.,0.,1.],[0.,3.,0.],[0.,0.,4.]])
    a=b+np.array([4.,1.,1.]);r=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    reference=tmp_path/'ref.pdb';global_move=tmp_path/'global.pdb';binder_move=tmp_path/'binder.pdb'
    _write_ca(reference,a,b)
    _write_ca(global_move,a@r+10,b@r+10)
    assert max(pose_geometry(reference,global_move).values())<1e-6
    _write_ca(binder_move,a+np.array([5.,0.,0.]),b)
    result=pose_geometry(reference,binder_move)
    assert result['binder_pose_ca_rmsd']==pytest.approx(5.)
    assert result['target_ca_rmsd']<1e-6 and result['binder_self_ca_rmsd']<1e-6
    from astrolabe.protein_beam_geometry import verify_prediction_sequences
    verify_prediction_sequences(binder_move,'AAAA','AAAA')
    with pytest.raises(ValueError,match='sequence'):
        verify_prediction_sequences(binder_move,'AAAD','AAAA')


def test_root_import_checks_scientific_identity_and_native_result(tmp_path):
    from omegaconf import OmegaConf
    from astrolabe.protein_beam_seeds import load_root
    from astrolabe.protein_beam_contract import content_sha256
    cfg=OmegaConf.create(dict(seqs=dict(A='AAA',B='+AAA'),types=dict(A='p',B='p'),homologies={},seed=5,
        loss=dict(loss_terms={}),evaluator=dict(alphafold3=dict(num_samples=1,service_socket=None))))
    snapshot=tmp_path/'scores.pkl';snapshot.write_bytes(b'native scores')
    structure=tmp_path/'model.cif';structure.write_text('native structure')
    c=row(0,(4.,.9,.7,-200.,2.));c.update(sequence='AAA',parent_id=None,seed=5,
        snapshot=str(snapshot),structure_path=str(structure))
    native=tmp_path/'candidates/000000/result.json';native.parent.mkdir(parents=True)
    native.write_text(json.dumps(c))
    contract=dict(config_yaml=OmegaConf.to_yaml(cfg),readonly_weight_asset_stat=[dict(size=50)],
        source_sha256={'interface_holes.py':'same','interaction.py':'same'},data_sha256={})
    state=tmp_path/'beam_state.json';state.write_text(json.dumps(dict(status='complete',budget_complete=True,
        config_hash='old',scientific_contract=contract,records=[c])))
    imported=load_root(cfg,state,contract)
    assert imported['evaluation_origin']=='reused_native'
    assert imported['imported_from']['snapshot_sha256']==content_sha256(snapshot)
    cfg.evaluator.alphafold3.num_samples=2
    with pytest.raises(ValueError,match='AF3'):load_root(cfg,state,contract)
    cfg.evaluator.alphafold3.num_samples=1
    c['metrics']['interface_buns']=0
    native.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='terminal'):load_root(cfg,state,contract)


def test_rejected_and_duplicate_cpu_proposals_resample_without_advancing_generation():
    prepared=prepare_one_generation(
        native_branch_config(),
        max_evaluations=3,
        proposals_per_parent=2,
        proposal_batch_size=2,
        pipeline=False,
    )
    def proposal(job,parent):
        if job.id==1:
            return dict(valid=False,proposal_rejections=['excluded_amino_acid'])
        sequence=list(str(parent['sequence']))
        if job.id in (2,3):
            sequence[0]='A'
        else:
            sequence[1]='A'
        return dict(sequence=''.join(sequence),valid=True)

    result=run_one_generation(prepared,proposal=proposal)

    assert result.state.generation==1
    assert result.state.proposals==4
    assert result.state.rejected==1
    assert result.state.duplicates==1
    assert result.state.seen[0]==prepared.parents[0]['sequence']
    assert tuple(member.id for member in result.state.records)==(0,1,2)
    assert tuple(record['id'] for record in result.proposals)==(1,2,3,4)
    assert tuple(record['id'] for record in result.evaluations)==(1,2)
    assert tuple(record['proposal_id'] for record in result.evaluations)==(2,4)
    assert result.state.counts==((0,2),)


def test_exhausted_proposal_cap_stays_bounded():
    prepared=prepare_one_generation(
        native_branch_config(),
        max_evaluations=3,
        proposals_per_parent=2,
        proposal_batch_size=3,
        pipeline=False,
    )
    result=run_one_generation(
        prepared,
        proposal=lambda job,parent:dict(
            valid=False,
            proposal_rejections=['total_mutation_limit'],
        ),
    )
    assert result.state.proposals==30
    assert result.state.rejected==30
    assert result.state.queued==() and result.state.pending==()
    assert tuple(member.id for member in result.state.records)==(0,)
    assert result.state.finished
    assert len(result.proposals)==30
    assert result.evaluations==()
