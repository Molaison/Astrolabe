"""Small real PDB/SciPy controls for lazily loaded spatial losses."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from astrolabe.losses import design_screening


@pytest.fixture
def interface_pdb(tmp_path):
    atoms = [
        ("CA", "A", 1, (0., 0., 0.), 80.),
        ("CB", "A", 1, (0., .5, 0.), 60.),
        ("CA", "A", 2, (0., 0., 1.), 40.),
        ("CA", "A", 3, (0., 0., 1.5), 20.),
        ("CA", "B", 1, (2., 0., 0.), 90.),
        ("CA", "B", 2, (10., 0., 0.), 50.),
    ]
    lines = [f"ATOM  {index:5d} {atom:^4s} ALA {chain}{residue:4d}    "
             f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00{bfactor:6.2f}           C"
             for index, (atom, chain, residue, xyz, bfactor) in enumerate(atoms, 1)]
    path = tmp_path / "interface.pdb"
    path.write_text("\n".join(lines + ["TER", "END", ""]))
    return path


@pytest.mark.parametrize("only_ca", [False, True])
def test_clash_count_retains_chain_and_sequence_exclusions(interface_pdb, only_ca):
    assert design_screening._calculate_bindcraft_clash_score(
        interface_pdb, threshold=2.4, only_ca=only_ca) == 3


def test_hotspots_and_key_residue_scores_retain_geometric_cutoff(interface_pdb):
    assert design_screening._bindcraft_hotspot_residues(
        interface_pdb, binder_chain="B", target_chain="A", atom_distance_cutoff=2.4) == {1: "A"}
    result = SimpleNamespace(names=["AB"], complex_paths={"AB": interface_pdb})
    assert design_screening.halludesign_key_res_plddt(
        result, protein_chain="A", ligand_chains="B", cutoff=2.4) == {"AB": 60.0}


def test_missing_spatial_dependency_retains_explicit_loss_error():
    script = """
import importlib.abc, sys
class MissingSpatial(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'scipy.spatial':
            raise ImportError('controlled missing scipy.spatial')
        return None
sys.meta_path.insert(0, MissingSpatial())
from astrolabe.losses import design_screening
try:
    design_screening._calculate_bindcraft_clash_score('must_not_read.pdb')
except ImportError as error:
    assert str(error) == 'scipy is required for BindCraft-aligned clash/interface-residue losses.'
else:
    raise AssertionError('missing dependency was not rejected')
"""
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parents[1],
                            text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
