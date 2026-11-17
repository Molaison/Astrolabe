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
    import subprocess
    import sys
    from pathlib import Path

    code = r'''
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from astrolabe.losses.interface_holes import native_interface_holes
import pyrosetta
from pathlib import Path
from astrolabe.utils.pyrosetta_init import pyrosetta_was_initialized
if not pyrosetta_was_initialized(pyrosetta):
    pyrosetta.init('-mute all -holes:dalphaball '+str(Path('astrolabe/assets/DAlphaBall.gcc').resolve()))
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from astrolabe.losses.independent import structure_context
fixture = Path(sys.argv[1]).parents[1] / "examples/paper/inputs/assemblies/01_C3_seed2503_step024_reference.cif"
atoms = structure_context(SimpleNamespace(complex_paths={"ABC": fixture}), "ABC")["atoms"]
pdb = PDBFile()
pdb.set_structure(atoms)
pose = pyrosetta.Pose()
pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose, str(pdb))
assert pose.total_residue() == struc.get_residue_count(atoms) == 216
value, selected = native_interface_holes(pose,['A'],['B', 'C'],10)
swapped, indices = native_interface_holes(pose,['B', 'C'],['A'],10)
assert np.isfinite(value) and value==pytest.approx(swapped, abs=1e-10)
assert selected==indices and selected
from astrolabe.losses.interface_holes import native_interface_holes_details
details=native_interface_holes_details(pose,['A'],['B', 'C'],10)
assert details['sum']==pytest.approx(value,abs=1e-9)
assert details['atom_count']>len(selected)
assert details['mean']==pytest.approx(value/details['atom_count'],abs=1e-12)


subset = selected[::2]
filtered, indices = native_interface_holes(
    pose, ["A"], ["B", "C"], selected_residues=subset)
details = native_interface_holes_details(
    pose, ["A"], ["B", "C"], selected_residues=subset)
assert indices == subset
assert details["sum"] == pytest.approx(filtered, abs=1e-9)
assert details["residue_count"] == len(subset)
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve())],
        capture_output=True, text=True, timeout=120,
        cwd=Path(__file__).parents[1],
    )
    assert result.returncode == 0, result.stdout + result.stderr
