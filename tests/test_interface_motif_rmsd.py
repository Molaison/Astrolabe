from types import SimpleNamespace

import numpy as np
import pytest
from Bio.PDB import Atom, Chain, Model, PDBIO, Residue, Structure

from astrolabe.losses.interface_motif import (
    interface_motif_rmsd, interface_motif_rmsd_norm, reference_interface_positions,
)
from astrolabe.losses.base import loss_registry


def write_structure(path, *, target=True, transform=False, changed=None,
                    distortion=None, missing=None, length=4, chain_id='A', target_offset=3.):
    structure = Structure.Structure('test'); model = Model.Model(0); structure.add(model)
    chain = Chain.Chain(chain_id); model.add(chain)
    for i in range(length):
        name = 'LYS' if i == changed else 'ALA'
        residue = Residue.Residue((' ', 101+i, ' '), name, ''); chain.add(residue)
        base = np.array([i*4., 0., (i % 2)*.6])
        for j, (atom, offset) in enumerate([('N', [-1.,0.,0.]), ('CA',[0.,.3,0.]), ('C',[1.,0.,0.]), ('O',[1.2,-.7,.3]), ('CB',[0.,1.4,.4])]):
            if (i, atom) == missing: continue
            point = base+offset
            if i == distortion: point = point + [0.,0.,2.]
            if transform: point = point @ np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]) + [11.,-7.,4.]
            residue.add(Atom.Atom(atom, np.array(point,dtype=float), 20.,1.,' ',f'{atom:>4}',i*5+j,element=atom[0]))
    if target:
        chain_b = Chain.Chain('B'); model.add(chain_b)
        # Reference contacts at separated positions 0 and 3; no middle contacts.
        for i, x in enumerate([0.,12.]):
            residue = Residue.Residue((' ', i+1,' '),'GLY',''); chain_b.add(residue)
            residue.add(Atom.Atom('CA',np.array([x,target_offset,0.]),20.,1.,' ',' CA ',100+i,element='C'))
    writer=PDBIO(); writer.set_structure(structure); writer.save(str(path))
    return path


@pytest.fixture
def case(tmp_path):
    ref=write_structure(tmp_path/'reference.pdb')
    pred=write_structure(tmp_path/'prediction.pdb',target=False)
    return ref,pred


def evaluate(ref,pred,**kwargs):
    result=SimpleNamespace(names=['A'],complex_paths={'A':str(pred)})
    params=dict(reference_complex=str(ref),binder_chain='A',reference_target_chains=['B'],contact_cutoff=2.)
    params.update(kwargs)
    return interface_motif_rmsd(result,**params)['A'],interface_motif_rmsd_norm(result,**params)['A']


def test_registry_and_reference_contact_positions(case):
    ref,pred=case
    assert reference_interface_positions(ref,'A',['B'],2.) == (0,3)
    assert 'interface_motif_rmsd' in loss_registry.valid_losses
    assert 'interface_motif_rmsd_norm' in loss_registry.valid_losses
    r,m=evaluate(ref,pred); assert r<1e-8 and m==pytest.approx(1.)


def test_joint_fit_is_rigid_body_invariant(case):
    ref,pred=case;write_structure(pred,target=False,transform=True)
    r,m=evaluate(ref,pred);assert r<1e-6 and m==pytest.approx(1.)


def test_relative_fragment_displacement_is_detected(case):
    ref,pred=case;write_structure(pred,target=False,distortion=3)
    r,m=evaluate(ref,pred);assert r>.1 and m==pytest.approx(max(0.,1.-r/2.)) and m<1.


def test_local_backbone_distortion_is_detected(case):
    ref,pred=case
    text=pred.read_text();lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('ATOM') and line[12:16].strip()=='O' and int(line[22:26])==101:
            lines[i]=line[:46]+f'{float(line[46:54])+1.5:8.3f}'+line[54:]
    pred.write_text('\n'.join(lines)+'\n')
    r,m=evaluate(ref,pred);assert r>.1 and m<1.


def test_outside_interface_substitution_does_not_change_mapping(case):
    ref,pred=case;write_structure(pred,target=False,changed=1)
    r,m=evaluate(ref,pred);assert r<1e-8 and m==pytest.approx(1.)


@pytest.mark.parametrize('change', ['interface_aa','backbone_atom','length','chain'])
def test_incompatible_prediction_rejected(case,change):
    ref,pred=case
    options={'interface_aa':{'changed':0},'backbone_atom':{'missing':(3,'O')},'length':{'length':3},'chain':{'chain_id':'C'}}
    write_structure(pred,target=False,**options[change])
    with pytest.raises(ValueError): evaluate(ref,pred)


@pytest.mark.parametrize('best,worst', [(0.,0.),(2.,1.),(-1.,2.),(float('nan'),2.),(0.,float('inf'))])
def test_invalid_normalization_rejected(case,best,worst):
    ref,pred=case;result=SimpleNamespace(names=['A'],complex_paths={'A':str(pred)})
    with pytest.raises(ValueError):interface_motif_rmsd_norm(result,str(ref),'A',best=best,worst=worst)


def test_missing_reference_target_and_no_contacts_rejected(case):
    ref,pred=case
    with pytest.raises(ValueError):reference_interface_positions(ref,'A',['C'],2.)
    with pytest.raises(ValueError):reference_interface_positions(ref,'A',['A'],2.)
    with pytest.raises(ValueError):reference_interface_positions(ref,'A',[],2.)
    with pytest.raises(ValueError):reference_interface_positions(ref,'A',['B'],.1)


@pytest.mark.parametrize('cutoff', [0.,-1.,float('nan'),float('inf')])
def test_invalid_contact_cutoff_rejected(case,cutoff):
    ref,_=case
    with pytest.raises(ValueError):reference_interface_positions(ref,'A',['B'],cutoff)


def test_normalization_clips_at_both_ends(case):
    ref,pred=case;write_structure(pred,target=False,distortion=3)
    result=SimpleNamespace(names=['A'],complex_paths={'A':str(pred)})
    kwargs=dict(reference_complex=str(ref),binder_chain='A',reference_target_chains=['B'],contact_cutoff=2.)
    assert interface_motif_rmsd_norm(result,**kwargs,best=0,worst=.01)['A']==0.
    assert interface_motif_rmsd_norm(result,**kwargs,best=10,worst=11)['A']==1.
