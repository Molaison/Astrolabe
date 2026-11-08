from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from astrolabe.mutators import rfd4_reconstruction as reconstruction
from astrolabe.mutators.rfd4_feedback import HBondObservation, PolarAtomObservation, ResidueObservation
from astrolabe.mutators.rfd4_types import AtomRef, ChainSpec, ResidueRef


def feedback(energy=-0.8):
    donor = AtomRef("A", 2, "ND2")
    acceptor = AtomRef("B", 1, "N9")
    polar = (
        PolarAtomObservation(donor, "ASN", "NH2O", "DONOR", (0., 0., 0.),
                             1., 2.8, False, True, True, 1),
        PolarAtomObservation(acceptor, "CFF", "Nhis", "ACCEPTOR", (2.8, 0., 0.),
                             0., 2.8, True, True, True, 1),
        PolarAtomObservation(AtomRef("B", 1, "O11"), "CFF", "O", "ACCEPTOR", (4., 1., 0.),
                             1., 3., False, True, False, 0),
        PolarAtomObservation(AtomRef("B", 1, "O13"), "CFF", "O", "ACCEPTOR", (4., 2., 0.),
                             2., 3., False, True, False, 0),
    )
    hb = HBondObservation(donor, AtomRef("A", 2, "HD21"), acceptor,
                          (0., 0., 0.), (1., 0., 0.), (2.8, 0., 0.),
                          energy, 1., True, True, True)
    return SimpleNamespace(
        binder_chains=("A",), polar_atoms=polar, interfacial_hbonds=(hb,),
        residues=(ResidueObservation(ResidueRef("A", 2), "ASN", 2, 2, "", -1., 1, 0),),
    )


@pytest.fixture
def source(tmp_path, monkeypatch):
    path = tmp_path / "parent.cif"
    path.write_bytes(b"fixture source identity")
    monkeypatch.setattr(reconstruction, "source_atom_names",
                        lambda p: {ResidueRef("A", 2): {"CG", "OD1", "ND2", "CA"}})
    return path


def plan(path, fb, **kwargs):
    return reconstruction.plan_reconstruction(
        path, (ChainSpec("A", "A", "ANA"), ChainSpec("B", "B", "CFF", "ligand")),
        fb, config={"length_min": 2, "length_max": 5}, **kwargs,
    )


def test_minimal_amide_frame_and_own_role_goal(source):
    result = plan(source, feedback())
    assert result.reconstruction["motifs"][0]["atoms"] == ["CG", "ND2", "OD1"]
    assert result.fixed_coordinates == (ResidueRef("B", 1),)
    assert result.released == tuple(ResidueRef("A", i) for i in range(1, 4))
    assert [(h.atom.atom, h.role) for h in result.hbonds] == [("O11", "ACCEPTOR")]
    reranked = plan(source, feedback(), failed_hbond_attempts={"B:1:O11": 1})
    assert reranked.hbonds[0].atom.atom == "O13"


def test_no_good_hbond_means_no_fabricated_packing_anchor(source):
    result = plan(source, feedback(energy=-0.4))
    assert result.reconstruction["motifs"] == []
    assert "de novo" in json.loads(result.feedback_json)["no_motif_behavior"]


def test_bad_angle_or_source_missing_is_not_a_retained_motif(source):
    fb = feedback()
    fb.interfacial_hbonds = (replace(fb.interfacial_hbonds[0], hydrogen_xyz=(-1., 0., 0.)),)
    assert plan(source, fb).reconstruction["motifs"] == []
    fb = feedback()
    fb.polar_atoms = (replace(fb.polar_atoms[0], source_present=False), *fb.polar_atoms[1:])
    assert plan(source, fb).reconstruction["motifs"] == []


def test_native_lineage_survives_changed_source_and_index(source, monkeypatch):
    monkeypatch.setattr(reconstruction, "read_motif_lineage", lambda p: {ResidueRef("A", 2): "ancestral-motif"})
    assert plan(source, feedback()).reconstruction["motifs"][0]["id"] == "ancestral-motif"
    source.write_bytes(b"new round coordinates and indices")
    assert plan(source, feedback()).reconstruction["motifs"][0]["id"] == "ancestral-motif"


def test_unknown_or_missing_functional_atom_fails(source, monkeypatch):
    with pytest.raises(ValueError, match="No functional geometry"):
        reconstruction.functional_atoms("ASN", "OXT")
    monkeypatch.setattr(reconstruction, "source_atom_names", lambda p: {ResidueRef("A", 2): {"ND2"}})
    with pytest.raises(ValueError, match="lacks functional"):
        plan(source, feedback())


@pytest.mark.parametrize("bounds", [
    {"length_min": True, "length_max": 5}, {"length_min": 5, "length_max": 4},
    {"length_min": 2, "length_max": 5, "unexpected": 1},
])
def test_invalid_length_declaration(bounds):
    with pytest.raises(ValueError):
        reconstruction.reconstruction_bounds(bounds)


def test_published_mapping_integrity_and_no_nearest_position_guess(tmp_path):
    structure = tmp_path / "rfd4.generated.cif"
    structure.write_bytes(b"native structure")
    sha = hashlib.sha256(structure.read_bytes()).hexdigest()
    motif = {"id": "persistent", "source": {"chain": "A", "position": 74}, "atoms": ["CG", "ND2", "OD1"]}
    mapped = {**motif, "output": {"chain": "A", "position": 18}}
    metadata_path = tmp_path / "rfd4.backend.json"
    index_path = tmp_path / "rfd4.rfd4.generated.json"

    def write(mapping):
        metadata_path.write_text(json.dumps({
            "checksums": {"generated.cif": sha}, "request": {"plan": {"reconstruction": {"motifs": [motif]}}},
            "validation": {"motif_mapping": mapping},
        }))
        index_path.write_text(json.dumps({"files": {
            structure.name: {"sha256": sha},
            metadata_path.name: {"sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()},
        }}))

    write([mapped])
    assert reconstruction.read_motif_lineage(structure) == {ResidueRef("A", 18): "persistent"}
    write([])
    with pytest.raises(ValueError, match="incomplete"):
        reconstruction.read_motif_lineage(structure)
    write([mapped])
    metadata_path.write_bytes(metadata_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="digest mismatch"):
        reconstruction.read_motif_lineage(structure)
