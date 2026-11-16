"""Behavioral probes for real independent libraries, without PyRosetta."""
from pathlib import Path

import numpy as np
import pytest

struc = pytest.importorskip("biotite.structure")
from astrolabe.losses import independent as metrics


def _atoms(records, bonds=()):
    atoms = struc.AtomArray(len(records))
    for field, column in (("chain_id", 0), ("res_id", 1), ("res_name", 2),
                          ("atom_name", 3), ("element", 4), ("coord", 5)):
        setattr(atoms, field, np.asarray([r[column] for r in records],
                                         dtype=float if field == "coord" else None))
    atoms.bonds = struc.BondList(len(atoms), np.asarray(bonds, dtype=int).reshape(-1, 3))
    return atoms


def _hbond_fixture():
    return _atoms([
        ("A", 1, "ASN", "ND2", "N", [0, 0, 0]),
        ("A", 1, "ASN", "HD21", "H", [1, 0, 0]),
        ("A", 1, "ASN", "CG", "C", [0, 1.3, 0]),
        ("B", 1, "ASP", "OD1", "O", [2.7, 0, 0]),
        ("B", 1, "ASP", "CG", "C", [3.9, 0, 0]),
    ], [(0, 1, 1), (0, 2, 1), (3, 4, 2)])


def _surface_fixture():
    return _atoms([
        (chain, i + 1, "ALA", "CA", "C", [x, y, z])
        for chain, x in (("A", 0), ("B", 3.8))
        for i, (y, z) in enumerate(((-2.5, -2.5), (-2.5, 0), (-2.5, 2.5),
                                   (0, -2.5), (0, 0), (0, 2.5),
                                   (2.5, -2.5), (2.5, 0), (2.5, 2.5)))
    ])


def _cms_fixture():
    source = Path(__file__).resolve().parents[1] / "examples/input/9TP_cid.cif"
    atoms = metrics.load_structure(source)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    a, b = atoms.chain_id == "A", atoms.chain_id == "B"
    from scipy.spatial.distance import cdist
    distances = cdist(atoms.coord[a], atoms.coord[b])
    i, j = np.unravel_index(distances.argmin(), distances.shape)
    midpoint = (atoms.coord[a][i] + atoms.coord[b][j]) / 2
    selected = np.zeros(len(atoms), dtype=bool)
    for side in (a, b):
        indices = np.flatnonzero(side)
        nearest = np.argsort(np.linalg.norm(atoms.coord[side] - midpoint, axis=1))[:24]
        selected[indices[nearest]] = True
    return atoms[selected]


def test_real_sc_matches_published_library_and_swapped_groups():
    library = pytest.importorskip("shape_complementarity")
    atoms = _surface_fixture()
    a, b = atoms.chain_id == "A", atoms.chain_id == "B"
    actual = metrics.shape_complementarity(atoms, a, b)
    expected = library.compute_sc(
        atoms.coord[a].tolist(), atoms.atom_name[a].tolist(), atoms.res_name[a].tolist(),
        atoms.coord[b].tolist(), atoms.atom_name[b].tolist(), atoms.res_name[b].tolist(),
        parallel=False,
    )
    assert actual.value == pytest.approx(expected.sc, abs=1e-12)
    assert metrics.shape_complementarity(atoms, b, a).value == pytest.approx(actual.value)
    assert actual.details["version"] != "unknown"


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.5])
@pytest.mark.parametrize("surface_side", ["a", "b"])
def test_cms_spatial_readout_matches_real_dense_reference(weight, surface_side):
    library = pytest.importorskip("py_contact_ms._core")
    atoms = _cms_fixture()
    a, b = atoms.chain_id == "A", atoms.chain_id == "B"
    calc = library.MolecularSurfaceCalculator()
    calc.settings.weight = weight
    radii = library.get_radii_from_names(atoms.res_name, atoms.atom_name)
    calc.add_binder_and_target(atoms.coord[a], radii[a], atoms.coord[b], radii[b])
    calc.CalcLoaded()
    # Native Rosetta's CalcContactArea integrates molecule 0 (selector1/A).
    expected, per_atom = calc.calc_contact_molecular_surface(target_side=surface_side == "b")
    actual = metrics.contact_molecular_surface(atoms, a, b, distance_weight=weight,
                                               surface_side=surface_side)
    assert actual.value == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert actual.details["per_target_atom"] == pytest.approx(per_atom, abs=1e-12)
    assert sum(actual.details["per_target_atom"]) == pytest.approx(actual.value)
    assert actual.details["surface_side"] == surface_side
    assert actual.details["target_atom_indices"] == np.flatnonzero(a if surface_side == "a" else b).tolist()


def test_cms_default_reports_a_and_preserves_asymmetry():
    atoms = _cms_fixture()
    a, b = atoms.chain_id == "A", atoms.chain_id == "B"
    default = metrics.contact_molecular_surface(atoms, a, b)
    side_b = metrics.contact_molecular_surface(atoms, a, b, surface_side="b")
    assert default.value == pytest.approx(60.01466349967167, rel=1e-12)
    assert side_b.value == pytest.approx(56.644147451794346, rel=1e-12)
    assert default.details["surface_side"] == "a"
    with pytest.raises(ValueError, match="surface_side"):
        metrics.contact_molecular_surface(atoms, a, b, surface_side="both")


@pytest.mark.parametrize("metric", [metrics.shape_complementarity, metrics.contact_molecular_surface])
def test_surface_radii_do_not_silently_guess_noncanonical_atoms(metric):
    atoms = _cms_fixture()
    a, b = atoms.chain_id == "A", atoms.chain_id == "B"
    atoms.res_name[b] = "LIG"
    with pytest.raises(metrics.BackendUnavailable, match="canonical"):
        metric(atoms, a, b)
    atoms = _cms_fixture()
    atoms.atom_name[np.flatnonzero(b)[0]] = "QX"
    with pytest.raises(metrics.BackendUnavailable, match="atom names"):
        metric(atoms, a, b)


@pytest.mark.parametrize("metric", [metrics.shape_complementarity, metrics.contact_molecular_surface])
def test_surface_atom_element_mismatch_is_invalid_input(metric):
    atoms = _surface_fixture()
    atoms.element[0] = "CA"
    with pytest.raises(ValueError, match="element"):
        metric(atoms, atoms.chain_id == "A", atoms.chain_id == "B")


def test_sc_rejects_hidden_radii_environment_override(tmp_path, monkeypatch):
    import json
    path = tmp_path / "radii.json"
    path.write_text(json.dumps([dict(residue="ALA", atom="CA", radius=0.1)]))
    atoms = _surface_fixture()
    for variable in ("ATOMIC_RADII", "ATOMIC_RADII_PATH"):
        with monkeypatch.context() as patch:
            patch.setenv(variable, str(path))
            with pytest.raises(ValueError, match="radii.*environment"):
                metrics.shape_complementarity(atoms, atoms.chain_id == "A", atoms.chain_id == "B")


@pytest.mark.parametrize("metric", [metrics.shape_complementarity, metrics.contact_molecular_surface])
def test_surface_selections_define_fragments_and_exclude_unselected_environment(metric):
    atoms = _cms_fixture()
    a, b = atoms.chain_id == "A", atoms.chain_id == "B"
    expected = metric(atoms, a, b).value
    # An overlapping spectator ligand is outside both selected surfaces.
    spectator = _atoms([("C", 1, "LIG", "CX", "C", atoms.coord[0].tolist())])
    extended = atoms + spectator
    actual = metric(extended, extended.chain_id == "A", extended.chain_id == "B")
    assert actual.value == pytest.approx(expected, abs=1e-12)
    assert "unselected atoms excluded" in actual.details["surface_scope"]


def test_surface_masks_validate_before_backend_import():
    atoms = _surface_fixture()
    a = atoms.chain_id == "A"
    with pytest.raises(ValueError, match="disjoint"):
        metrics.shape_complementarity(atoms, a, a)
    with pytest.raises(ValueError, match="boolean"):
        metrics.shape_complementarity(atoms, [0, 1], ~a)
    atoms.coord[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        metrics.contact_molecular_surface(atoms, a, ~a)


def test_explicit_rosetta_radii_are_never_ignored():
    atoms = _surface_fixture()
    a, b = atoms.chain_id == "A", atoms.chain_id == "B"
    with pytest.raises(metrics.BackendUnavailable, match="Rosetta LJ radii"):
        metrics.contact_molecular_surface(atoms, a, b, use_rosetta_radii=True)


def test_cms_separated_groups_have_measured_zero_area():
    atoms = _cms_fixture()
    a, b = atoms.chain_id == "A", atoms.chain_id == "B"
    atoms.coord[b] += 100
    result = metrics.contact_molecular_surface(atoms, a, b)
    assert result.value == 0.0
    assert result.details["empty_interface"] is True
    assert np.all(np.asarray(result.details["per_target_atom"]) == 0)


def test_cms_handles_small_regions_without_convex_or_toroid_queue():
    atoms = _atoms([
        ("A", 1, "ALA", "CA", "C", [0, 0, 0]),
        ("A", 2, "ALA", "CA", "C", [0, 0, 3]),
        ("B", 1, "ALA", "CA", "C", [2.8, 0, 0]),
        ("B", 2, "ALA", "CA", "C", [2.8, 0, 3]),
    ])
    a, b = atoms.chain_id == "A", atoms.chain_id == "B"
    assert metrics.contact_molecular_surface(atoms, a, b).value > 0
    grid = _surface_fixture()
    result = metrics.contact_molecular_surface(grid, grid.chain_id == "A", grid.chain_id == "B")
    assert np.isfinite(result.value)
    assert result.value >= 0


def test_real_hbond_geometry_direction_and_chemical_typing():
    atoms = _hbond_fixture()
    a, b = atoms.chain_id == "A", atoms.chain_id == "B"
    assert metrics.hbond_count(atoms, a, b, hydrogen_policy="require").value == 1
    assert metrics.hbond_count(atoms, b, a, hydrogen_policy="require").value == 1
    atoms.coord[1] = [0, 1, 0]
    assert metrics.hbond_count(atoms, a, b, hydrogen_policy="require").value == 0
    atoms.coord[1] = [1, 0, 0]
    atoms.res_name[3], atoms.atom_name[3], atoms.element[3] = "PRO", "N", "N"
    # Amide N is not an acceptor, even with perfect donor-H..N geometry.
    assert metrics.hbond_count(atoms, a, b, hydrogen_policy="require").value == 0


def test_protonated_carboxyl_hydroxyl_is_not_an_acceptor():
    atoms = _hbond_fixture() + _atoms([("B", 1, "ASP", "HD1", "H", [2.7, 1, 0])])
    atoms.bonds.add_bond(3, len(atoms) - 1, 1)
    assert metrics.hbond_count(atoms, atoms.chain_id == "A", atoms.chain_id == "B",
                                hydrogen_policy="require").value == 0


def test_partial_hydrogen_coverage_does_not_silently_hide_a_donor():
    atoms = _hbond_fixture()
    atoms.res_name[3], atoms.atom_name[3], atoms.element[3] = "LYS", "NZ", "N"
    with pytest.raises(ValueError, match="coverage is incomplete"):
        metrics.hbond_count(atoms, atoms.chain_id == "A", atoms.chain_id == "B", hydrogen_policy="require")


def test_missing_hydrogens_fail_instead_of_returning_zero():
    atoms = _hbond_fixture()
    atoms = atoms[atoms.element != "H"]
    with pytest.raises(ValueError, match="explicit hydrogen"):
        metrics.hbond_count(atoms, atoms.chain_id == "A", atoms.chain_id == "B",
                            hydrogen_policy="require")


def test_selected_hbond_regions_are_respected():
    atoms = _hbond_fixture()
    a, b = atoms.chain_id == "A", atoms.chain_id == "B"
    b[3] = False
    assert metrics.hbond_count(atoms, a, b, hydrogen_policy="require").value == 0


def _buried_acceptor_fixture():
    records = [
        ("A", 1, "ALA", "O", "O", [0, 0, 0]),
        ("A", 1, "ALA", "C", "C", [-1.2, 0, 0]),
        ("A", 2, "ALA", "N", "N", [10, 0, 0]),
        ("A", 2, "ALA", "H", "H", [11, 0, 0]),
    ]
    records += [("B", i + 1, "ALA", "CA", "C", xyz) for i, xyz in enumerate(
        ([2.8, 0, 0], [-2.8, 0, 0], [0, 2.8, 0], [0, -2.8, 0], [0, 0, 2.8], [0, 0, -2.8]))]
    return _atoms(records, [(0, 1, 2), (2, 3, 1)])


def test_real_buns_detects_interface_burial_and_retains_full_environment():
    atoms = _buried_acceptor_fixture()
    a, b = atoms.chain_id == "A", atoms.chain_id == "B"
    actual = metrics.interface_buns(atoms, a, b, hydrogen_policy="require")
    assert actual.value == 1
    assert actual.details["bound_buns"] == 1
    assert actual.details["separated_buns"] == 0
    assert metrics.interface_buns(atoms, b, a, hydrogen_policy="require").value == 1
    region_b = b.copy()
    region_b[5:] = False
    assert metrics.interface_buns(atoms, a, region_b, group_a=a, group_b=b,
                                  hydrogen_policy="require").value == 1
    region_a = a.copy()
    region_a[0] = False
    assert metrics.interface_buns(atoms, region_a, b, group_a=a, group_b=b,
                                  hydrogen_policy="require").value == 0


def test_real_hydride_preparation_is_shared_and_content_bound():
    pytest.importorskip("hydride")
    source = Path(__file__).resolve().parents[1] / "examples/input/9TP_cid.cif"
    atoms = metrics.load_structure(source)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    first = metrics._hydrogenated(atoms, "add", 7.0)
    assert first is metrics._hydrogenated(atoms.copy(), "add", 7.0)
    np.testing.assert_allclose(first[0].coord[first[1] >= 0], atoms.coord)
    moved = atoms.copy()
    moved.coord[0, 0] += 0.1
    assert metrics._hydrogenated(moved, "add", 7.0) is not first


def test_file_reload_binds_bytes_and_returns_isolated_arrays(tmp_path):
    from biotite.structure.io.pdb import PDBFile
    atoms = _hbond_fixture()
    path = tmp_path / "complex.pdb"
    file = PDBFile()
    file.set_structure(atoms)
    file.write(path)
    first = metrics.load_structure(path)
    first.coord[0, 0] = 100
    assert metrics.load_structure(path).coord[0, 0] == 0
    atoms.coord[0, 0] = 2
    file = PDBFile()
    file.set_structure(atoms)
    file.write(path)
    assert metrics.load_structure(path).coord[0, 0] == 2


def test_mmcif_uses_evaluator_label_chains_and_preserves_insertion_codes(tmp_path):
    from biotite.structure.io import pdbx
    atoms = _hbond_fixture()
    atoms.ins_code[:3] = "I"
    file = pdbx.CIFFile()
    pdbx.set_structure(file, atoms)
    category = file.block["atom_site"]
    category["auth_asym_id"] = pdbx.CIFColumn(np.where(atoms.chain_id == "A", "X", "Y"))
    path = tmp_path / "chain_namespaces.cif"
    file.write(path)
    loaded = metrics.load_structure(path)
    assert loaded.chain_id.tolist() == ["A", "A", "A", "B", "B"]
    assert loaded.ins_code.tolist() == ["I", "I", "I", "", ""]


def test_mmcif_nonpolymer_residues_keep_distinct_author_numbers(tmp_path):
    from biotite.structure.io import pdbx
    atoms = _atoms([
        ("W", 10, "HOH", "O", "O", [0, 0, 0]),
        ("W", 20, "HOH", "O", "O", [10, 0, 0]),
    ])
    atoms.hetero[:] = True
    file = pdbx.CIFFile()
    pdbx.set_structure(file, atoms)
    category = file.block["atom_site"]
    category["label_asym_id"] = pdbx.CIFColumn(["W", "W"])
    category["label_seq_id"] = pdbx.CIFColumn([".", "."])
    category["auth_asym_id"] = pdbx.CIFColumn(["Q", "Q"])
    path = tmp_path / "nonpolymer.cif"
    file.write(path)
    loaded = metrics.load_structure(path)
    assert loaded.chain_id.tolist() == ["W", "W"]
    assert loaded.res_id.tolist() == [10, 20]


def test_unsupported_chemistry_is_capability_absence_but_bad_geometry_is_not():
    atoms = _hbond_fixture()
    a, b = atoms.chain_id == "A", atoms.chain_id == "B"
    atoms.res_name[b] = "LIG"
    with pytest.raises(metrics.BackendUnavailable, match="unsupported residues"):
        metrics.hbond_count(atoms, a, b, hydrogen_policy="require")
    atoms.coord[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        metrics.hbond_count(atoms, a, b, hydrogen_policy="require")


@pytest.mark.parametrize("options", [
    {"cutoff_distance": -1}, {"cutoff_angle": 181}, {"ph": float("nan")},
    {"hydrogen_policy": "ignore"},
])
def test_invalid_hbond_options_fail(options):
    atoms = _hbond_fixture()
    with pytest.raises(ValueError):
        metrics.hbond_count(atoms, atoms.chain_id == "A", atoms.chain_id == "B", **options)
