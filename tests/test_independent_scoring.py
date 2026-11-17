from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("biotite")
pytest.importorskip("hydride")
pytest.importorskip("rdkit")

from astrolabe.losses.independent import (
    buried_unsatisfied_count, crossgroup_hbond_count, hydrogen_bonds,
    independent_cms, independent_sc, interface_scores, polar_context,
    structure_context, validate_backend,
)

FIXTURE = Path(__file__).parents[1] / "examples/paper/inputs/assemblies/01_C3_seed2503_step024_reference.cif"


@pytest.fixture(scope="module")
def real_context():
    result = SimpleNamespace(names=["ABC"], complex_paths={"ABC": FIXTURE})
    return result, structure_context(result, "ABC")


def test_content_cache_observes_same_path_replacement(tmp_path):
    target = tmp_path / "model.cif"
    target.write_bytes(FIXTURE.read_bytes())
    result = SimpleNamespace(names=["ABC"], complex_paths={"ABC": target})
    first = structure_context(result, "ABC")
    assert structure_context(result, "ABC") is first
    target.write_bytes(target.read_bytes() + b"\n# changed artifact\n")
    assert structure_context(result, "ABC") is not first


def test_real_independent_surfaces_are_finite_and_separation_changes_cms(real_context):
    pytest.importorskip("shape_complementarity")
    pytest.importorskip("py_contact_ms")
    _result, context = real_context
    atoms = context["atoms"]
    a, b = atoms.chain_id == "A", np.isin(atoms.chain_id, ["B", "C"])
    sc = independent_sc(atoms, a, b)
    cms = independent_cms(atoms, a, b)
    assert -1 <= sc <= 1
    assert cms > 0
    shifted = atoms.copy()
    shifted.coord[b] += np.array([500.0, 0.0, 0.0])
    assert independent_cms(shifted, a, b) == pytest.approx(0.0, abs=1e-8)


def test_empty_surface_is_not_silently_scored_as_valid(real_context):
    pytest.importorskip("shape_complementarity")
    result, _context = real_context
    with pytest.raises(ValueError, match="no heavy atoms"):
        interface_scores(result, "sc", "A", ["B", "C"], selector_a="False")


def test_preparation_preserves_heavy_coordinates_and_rng(real_context):
    _result, context = real_context
    original = context["atoms"].coord.copy()
    state = np.random.get_state()
    prepared = polar_context(context)
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    np.testing.assert_array_equal(context["atoms"].coord, original)
    assert polar_context(context) is prepared
    assert len(prepared["atoms"]) > len(context["atoms"])


def _explicit_molecules(smiles):
    import biotite.structure as struc
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=11) == 0
    atoms = struc.AtomArray(molecule.GetNumAtoms())
    atoms.coord = molecule.GetConformer().GetPositions()
    atoms.element = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    atoms.atom_name = [f"A{i}" for i in range(len(atoms))]
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 1
    atoms.res_name[:] = "UNK"
    atoms.set_annotation("charge", np.array([atom.GetFormalCharge() for atom in molecule.GetAtoms()]))
    orders = {Chem.BondType.SINGLE: struc.BondType.SINGLE,
              Chem.BondType.DOUBLE: struc.BondType.DOUBLE,
              Chem.BondType.AROMATIC: struc.BondType.AROMATIC_SINGLE}
    atoms.bonds = struc.BondList(len(atoms), np.array([
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), orders[bond.GetBondType()])
        for bond in molecule.GetBonds()
    ], dtype=np.uint32))
    return {"atoms": atoms, "polar": {}}


def test_acceptor_typing_excludes_amide_and_positive_nitrogen():
    prepared = polar_context(_explicit_molecules("NC=O.C[NH3+]"), hydrogens="existing")
    atoms = prepared["atoms"]
    assert not np.any(prepared["acceptors"][atoms.element == "N"])
    assert np.any(prepared["acceptors"][atoms.element == "O"])
    assert np.any(prepared["donors"][atoms.element == "N"])


def _aligned_hbond():
    context = _explicit_molecules("CN.C=O")
    atoms = context["atoms"]
    donor = int(np.flatnonzero(atoms.element == "N")[0])
    acceptor = int(np.flatnonzero(atoms.element == "O")[0])
    neighbors, _orders = atoms.bonds.get_bonds(donor)
    hydrogen = int(next(i for i in neighbors if atoms.element[i] == "H"))
    atoms.coord[:] = np.arange(len(atoms))[:, None] * 10 + [100.0, 100.0, 100.0]
    atoms.coord[donor] = [0, 0, 0]
    atoms.coord[hydrogen] = [1, 0, 0]
    atoms.coord[acceptor] = [3, 0, 0]
    return context, donor, hydrogen, acceptor


def test_hbond_requires_angle_and_acceptor_chemistry():
    context, donor, hydrogen, acceptor = _aligned_hbond()
    prepared = polar_context(context, hydrogens="existing")
    assert [donor, hydrogen, acceptor] in hydrogen_bonds(prepared).tolist()
    bent, _, _, _ = _aligned_hbond()
    bent["atoms"].coord[acceptor] = [1, 2, 0]
    bad = polar_context(bent, hydrogens="existing")
    assert [donor, hydrogen, acceptor] not in hydrogen_bonds(bad).tolist()


def test_intrachain_hbond_satisfies_selected_polar_atom():
    context, donor, hydrogen, acceptor = _aligned_hbond()
    prepared = polar_context(context, hydrogens="existing")
    selected = np.arange(len(prepared["atoms"])) == acceptor
    assert buried_unsatisfied_count(prepared, selected, buried_sasa_cutoff=10000, sasa_points=100) == 0
    empty = np.zeros(len(prepared["atoms"]), dtype=bool)
    assert crossgroup_hbond_count(prepared, selected, empty) == 0
    assert buried_unsatisfied_count(prepared, empty, sasa_points=100) == 0


def test_missing_explicit_hydrogens_is_an_error(real_context):
    _result, context = real_context
    with pytest.raises(ValueError, match="explicit hydrogen"):
        polar_context(context, hydrogens="existing")


@pytest.mark.parametrize("options", [
    {"hydrogens": "guess"}, {"hydrogen_ph": float("nan")}, {"hydrogen_ph": True},
    {"hydrogen_ph": -1}, {"hydrogen_ph": 15},
])
def test_invalid_hydrogen_controls_fail(real_context, options):
    _result, context = real_context
    with pytest.raises(ValueError):
        polar_context(context, **options)


@pytest.mark.parametrize("value", ["auto", None, "unknown"])
def test_backend_is_explicit_and_deterministic(value):
    with pytest.raises(ValueError):
        validate_backend(value, "sc_rs")


def test_independent_cms_rejects_unsupported_radii(real_context):
    pytest.importorskip("py_contact_ms")
    _result, context = real_context
    atoms = context["atoms"].copy()
    atoms.atom_name[0] = "ZZZ"
    with pytest.raises(ValueError, match="no valid"):
        independent_cms(atoms, atoms.chain_id == "A", atoms.chain_id == "B")
