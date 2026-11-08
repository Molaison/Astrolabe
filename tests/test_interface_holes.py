from types import SimpleNamespace
import math
import numpy as np
import pytest
from astrolabe.losses.interface_holes import interface_residues, native_interface_holes


class Pose:
    def __init__(self, coords, chains):
        self.coords, self.chains = coords, chains
    def total_residue(self): return len(self.coords)
    def pdb_info(self): return SimpleNamespace(chain=lambda i:self.chains[i-1])
    def residue(self, i):
        return SimpleNamespace(is_protein=lambda:True, nheavyatoms=lambda:1,
                               xyz=lambda j:SimpleNamespace(x=self.coords[i-1][0],y=0.,z=0.))


def test_two_sides_and_full_pose_environment():
    pose = Pose([[0],[30],[5],[100]], ['A','A','B','B'])
    assert interface_residues(pose,['A'],['B'],10)==[1,3]
    assert interface_residues(pose,['B'],['A'],10)==[1,3]


@pytest.mark.parametrize('a,b,d', [(['A'],['A'],10),(['A'],['C'],10),(['A'],['B'],0),(['A'],['B'],float('nan'))])
def test_invalid_groups_or_distance(a,b,d):
    with pytest.raises(ValueError): interface_residues(Pose([[0],[5]],['A','B']),a,b,d)


def test_empty_interface_and_nonfinite_coordinates():
    with pytest.raises(ValueError): interface_residues(Pose([[0],[20]],['A','B']),['A'],['B'],10)
    with pytest.raises(ValueError): interface_residues(Pose([[0],[math.nan]],['A','B']),['A'],['B'],10)


def test_real_native_score_is_one_complete_pose_call(monkeypatch):
    import pyrosetta
    from pyrosetta.rosetta.protocols import simple_filters
    pose = Pose([[0],[30],[5],[100]], ['A','A','B','B'])
    calls=[]
    class FakeFilter:
        def set_residue_selector(self,s): calls.append('selector')
        def report_sm(self,p):
            assert p is pose
            calls.append('report')
            return -0.5
    monkeypatch.setattr(simple_filters,'HolesFilter',FakeFilter)
    value, selected = native_interface_holes(pose,['A'],['B'],10)
    assert value==-0.5 and selected==[1,3] and calls==['selector','report']


def test_real_fixture_holes():
    import pyrosetta
    from pathlib import Path
    from astrolabe.utils.pyrosetta_init import pyrosetta_was_initialized
    if not pyrosetta_was_initialized(pyrosetta):
        pyrosetta.init('-mute all -holes:dalphaball '+str(Path('astrolabe/assets/DAlphaBall.gcc').resolve()))
    fixture=Path('/case/config/001_binder_interface/01_astrolabe_beam/02_implement_shared_beam/03_fixture.pdb')
    if not fixture.exists(): pytest.skip('Engineering fixture not mounted')
    pose=pyrosetta.pose_from_file(str(fixture))
    value, selected = native_interface_holes(pose,['A'],['B'],10)
    swapped, indices = native_interface_holes(pose,['B'],['A'],10)
    assert np.isfinite(value) and value==pytest.approx(swapped, abs=1e-10)
    assert selected==indices and selected
    from astrolabe.losses.interface_holes import native_interface_holes_details
    details=native_interface_holes_details(pose,['A'],['B'],10)
    assert details['sum']==pytest.approx(value,abs=1e-9)
    assert details['atom_count']>len(selected)
    assert details['mean']==pytest.approx(value/details['atom_count'],abs=1e-12)
