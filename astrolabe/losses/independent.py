"""Independent structural measurements; never imports or initializes PyRosetta.

Surface scores use the published libraries' SC radii. Hydrogen-bond and BUNS
scores use named geometric definitions, not Rosetta energy thresholds.
RosettaHoles has no equivalent implementation here: a cavity proxy is not Holes.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import gzip
import hashlib
import importlib
import importlib.metadata
import io
import math
import os
from pathlib import Path
import threading
from typing import Any
import warnings

import numpy as np


class BackendUnavailable(ImportError):
    """The requested optional implementation/capability is not installed."""


@dataclass(frozen=True)
class MetricResult:
    value: float
    method: str
    details: dict[str, Any]


def _module(name: str, distribution: str | None = None):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise BackendUnavailable(
            f"Independent scoring requires {distribution or name}; "
            "install Astrolabe[structure-metrics] or select backend='rosetta'"
        ) from exc


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _result(value, method, *, backend, units, definition, **details):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{method} returned a nonfinite measurement")
    return MetricResult(value, method, {
        "backend": backend, "version": _version(backend), "units": units,
        "definition": definition, **details,
    })


def _validate_atoms(atoms):
    if len(atoms) == 0 or np.asarray(atoms.coord).shape != (len(atoms), 3):
        raise ValueError("Independent scoring requires a nonempty single structure")
    if not np.isfinite(atoms.coord).all():
        raise ValueError("Independent scoring requires finite atom coordinates")
    identities = list(zip(atoms.chain_id, atoms.res_id, atoms.ins_code,
                          atoms.res_name, atoms.atom_name))
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate atom identities: resolve alternate locations before scoring")
    return atoms


@lru_cache(maxsize=8)
def _read_structure(content: bytes, suffix: str):
    # Content, not path or mtime, owns cache identity; returned arrays are copied.
    if suffix in {".cif", ".mmcif"}:
        pdbx = _module("biotite.structure.io.pdbx", "biotite")
        parsed = pdbx.CIFFile.read(io.StringIO(content.decode("utf-8")))
        category = parsed.block["atom_site"]
        # Chain identities come from evaluator label_asym_id. Preserve author
        # residue numbers: non-polymers may have '.' for every label_seq_id.
        if "label_asym_id" in category:
            category["auth_asym_id"] = category["label_asym_id"]
        atoms = pdbx.get_structure(
            parsed,
            model=1, altloc="first", extra_fields=["charge"], include_bonds=True,
            use_author_fields=True,
        )
    elif suffix in {".pdb", ".ent"}:
        pdb = _module("biotite.structure.io.pdb", "biotite")
        atoms = pdb.PDBFile.read(io.StringIO(content.decode("utf-8"))).get_structure(
            model=1, altloc="first", extra_fields=["charge"], include_bonds=True,
        )
    else:
        raise ValueError(f"Unsupported structure format {suffix!r}; use PDB or mmCIF")
    return _validate_atoms(atoms)


def _pose_atoms(pose):
    """Read an already-owned pose, including its current coordinates and bonds."""
    struc = _module("biotite.structure", "biotite")
    records, atom_indices, bonds = [], {}, []
    pdb = pose.pdb_info()
    for ri in range(1, pose.total_residue() + 1):
        res = pose.residue(ri)
        for ai in range(1, res.natoms() + 1):
            if res.is_virtual(ai):
                continue
            xyz = res.xyz(ai)
            atom_indices[ri, ai] = len(records)
            records.append((str(pdb.chain(ri)), int(pdb.number(ri)),
                            str(pdb.icode(ri)).strip(), str(res.name3()).strip(),
                            str(res.atom_name(ai)).strip(),
                            str(res.atom_type(ai).element()).strip().upper(),
                            (xyz.x, xyz.y, xyz.z), not res.is_protein()))
    if not records:
        raise ValueError("The source pose contains no physical atoms")
    atoms = struc.AtomArray(len(records))
    for field, column in (("chain_id", 0), ("res_id", 1), ("ins_code", 2),
                          ("res_name", 3), ("atom_name", 4), ("element", 5),
                          ("coord", 6), ("hetero", 7)):
        setattr(atoms, field, np.asarray([record[column] for record in records]))
    # CCD supplies bond orders; preserve extra cyclic/disulfide pose connections.
    atoms.bonds = struc.connect_via_residue_names(atoms)
    known = {tuple(sorted(pair[:2])) for pair in atoms.bonds.as_array()}
    for (ri, ai), index in atom_indices.items():
        res = pose.residue(ri)
        for neighbor in res.bonded_neighbor(ai):
            other = atom_indices.get((ri, int(neighbor)))
            if other is not None and tuple(sorted((index, other))) not in known:
                bonds.append((index, other, int(struc.BondType.SINGLE)))
        for connection in range(1, res.n_possible_residue_connections() + 1):
            if res.residue_connection(connection).atomno() != ai:
                continue
            partner = res.connected_residue_at_resconn(connection)
            if not partner:
                continue
            partner_conn = res.residue_connection_conn_id(connection)
            other_atom = pose.residue(partner).residue_connection(partner_conn).atomno()
            other = atom_indices.get((partner, other_atom))
            if other is not None and tuple(sorted((index, other))) not in known:
                bonds.append((index, other, int(struc.BondType.SINGLE)))
    if bonds:
        atoms.bonds = atoms.bonds.merge(struc.BondList(len(atoms), np.asarray(bonds)))
    return _validate_atoms(atoms)


def load_structure(source):
    """Load PDB/mmCIF or copy an AtomArray/current pose, never a stale pose path."""
    struc = _module("biotite.structure", "biotite")
    if isinstance(source, struc.AtomArray):
        return _validate_atoms(source.copy())
    if hasattr(source, "total_residue") and hasattr(source, "residue"):
        return _pose_atoms(source)
    path = Path(source)
    content = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix in {".gz", ".gzip"}:
        content, suffix = gzip.decompress(content), Path(path.stem).suffix.lower()
    return _read_structure(content, suffix).copy()


def _mask(atoms, selection, label):
    result = np.asarray(selection)
    if result.dtype != np.bool_ or result.shape != (len(atoms),):
        raise ValueError(f"{label} must be a full-length boolean atom mask")
    return result


def _sides(atoms, mask_a, mask_b):
    _validate_atoms(atoms)
    a, b = _mask(atoms, mask_a, "mask_a"), _mask(atoms, mask_b, "mask_b")
    if np.any(a & b) or not a.any() or not b.any():
        raise ValueError("Scoring sides must be nonempty and disjoint")
    heavy = ~np.isin(atoms.element, ["H", "D"])
    if not np.any(a & heavy) or not np.any(b & heavy):
        raise ValueError("Each scoring side must contain heavy atoms")
    return a, b


@lru_cache(maxsize=20)
def _canonical_atom_elements(residue_name):
    info = _module("biotite.structure.info", "biotite")
    residue = info.residue(residue_name)
    return dict(zip(residue.atom_name.tolist(), residue.element.tolist()))


def _validate_surface_chemistry(atoms, selection):
    """Do not let name-based radius tables silently guess ligand/metal radii."""
    selected = atoms[selection & ~np.isin(atoms.element, ["H", "D"])]
    unsupported = set(selected.res_name) - _CANONICAL
    if unsupported:
        raise BackendUnavailable(
            "Independent surfaces support validated canonical amino-acid atom types; "
            f"unsupported residues: {sorted(unsupported)}. Select the Rosetta backend."
        )
    for residue_name in np.unique(selected.res_name):
        known = _canonical_atom_elements(str(residue_name))
        residue = selected[selected.res_name == residue_name]
        unknown = set(residue.atom_name) - known.keys()
        if unknown:
            raise BackendUnavailable(
                f"Independent surfaces have unsupported atom names in {residue_name}: {sorted(unknown)}"
            )
        for name, element in zip(residue.atom_name, residue.element):
            if element != known[name]:
                raise ValueError(
                    f"Atom element {element!r} does not match {residue_name}:{name} ({known[name]})"
                )


def shape_complementarity(atoms, mask_a, mask_b, *, parallel=False):
    """Lawrence-Colman SC from the independent sc-rs implementation."""
    a, b = _sides(atoms, mask_a, mask_b)
    _validate_surface_chemistry(atoms, a | b)
    if any(name in os.environ for name in ("ATOMIC_RADII", "ATOMIC_RADII_PATH")):
        raise ValueError(
            "Independent SC requires bundled radii; remove the ATOMIC_RADII/ATOMIC_RADII_PATH "
            "environment override before scoring"
        )
    module = _module("shape_complementarity", "shape-complementarity")
    heavy = ~np.isin(atoms.element, ["H", "D"])
    a, b = atoms[a & heavy], atoms[b & heavy]
    measured = module.compute_sc(
        a.coord.tolist(), a.atom_name.tolist(), a.res_name.tolist(),
        b.coord.tolist(), b.atom_name.tolist(), b.res_name.tolist(), parallel=parallel,
    )
    if not -1.0 <= measured.sc <= 1.0:
        raise ValueError("Independent SC is outside its mathematical range [-1, 1]")
    if not all(math.isfinite(float(v)) for v in (measured.trimmed_area, measured.median_distance)):
        raise ValueError("Independent SC returned nonfinite surface details")
    return _result(
        measured.sc, "lawrence_colman_sc/sc-rs", backend="shape-complementarity",
        units="dimensionless", definition="Lawrence-Colman surface-normal statistic",
        radii="sc-rs bundled SC radii", trimmed_area=float(measured.trimmed_area),
        median_distance=float(measured.median_distance),
        atoms_a=int(measured.atoms_a), atoms_b=int(measured.atoms_b),
        surface_scope="selected canonical amino-acid atoms; unselected atoms excluded",
    )


def contact_molecular_surface(
    atoms, mask_a, mask_b, *, distance_weight=0.5, use_rosetta_radii=None,
    surface_side="a",
):
    """Distance-weighted selected-side area; default A matches Rosetta selector1."""
    a, b = _sides(atoms, mask_a, mask_b)
    if surface_side not in {"a", "b"}:
        raise ValueError("surface_side must be 'a' or 'b'")
    weight = float(distance_weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("distance_weight must be finite and nonnegative")
    if use_rosetta_radii is True:
        raise BackendUnavailable("py-contact-ms uses SC radii, not Rosetta LJ radii")
    if use_rosetta_radii not in (None, False):
        raise ValueError("use_rosetta_radii must be None or a boolean")
    _validate_surface_chemistry(atoms, a | b)
    # This pinned adapter uses the 0.1.1 calculator and validates its readout
    # against the library's original dense nearest-distance implementation.
    module = _module("py_contact_ms._core", "py-contact-ms")
    from scipy.spatial import cKDTree

    class Surface(module.MolecularSurfaceCalculator):
        def calc_dots_for_all_atoms(self, _atoms_unused):
            # Keep the upstream phases and attention masks, but do not index
            # an AtomArray with an empty Python list (0.1.1 treats it as float).
            atoms = self.run.atoms
            self.run.radmax = atoms.radius.max()
            good = np.flatnonzero(self.build_neighbor_arrays())
            self.second_loop(atoms[good])
            attention = atoms.atten[good]
            convex = good[
                atoms.access[good].astype(bool)
                & (attention > module.ATTEN_BLOCKER)
                & ~((attention == module.ATTEN_6)
                    & (self.run.buried_array.nneighbors[good] == 0))
            ]
            if len(convex):
                self.generate_convex_surface(atoms[convex])
            if self.run.toroid_queue:
                self.generate_toroidal_surfaces()
            self.run.probes.finalize()
            if self.settings.rp > 0:
                self.generate_concave_surface()
            return 1

        # Bound temporary tensors without changing per-pair geometry or sampling.
        def vec_third_loop(self, atom1, atom2, uij, tij, rij):
            access = []
            for start in range(0, len(atom1), 128):
                part = slice(start, start + 128)
                access.append(super().vec_third_loop(
                    atom1[part], atom2[part], uij[part], tij[part], rij[part]))
            return np.unique(np.concatenate(access)) if access else np.empty(0, dtype=int)

        def generate_toroidal_surface(self, atom1, atom2, uij, tij, rij, between):
            access_a, access_b = [], []
            for start in range(0, len(atom1), 128):
                part = slice(start, start + 128)
                a, b = super().generate_toroidal_surface(
                    atom1[part], atom2[part], uij[part], tij[part], rij[part], between[part])
                access_a.append(a)
                access_b.append(b)
            return (np.concatenate(access_a), np.concatenate(access_b)) if access_a else (
                np.empty(0, dtype=bool), np.empty(0, dtype=bool))

        def calc_contact_molecular_surface(self, target_side=True):
            query, reference = (1, 0) if target_side else (0, 1)
            group = self.run.atoms.molecule[:len(self.run.atoms)]
            count = int(np.count_nonzero(group == query))
            offset = int(np.count_nonzero(group == 0)) if query else 0
            q, r = self.run.dots[query], self.run.dots[reference]
            q_buried, r_buried = q.buried.astype(bool), r.buried.astype(bool)
            if not np.any(q_buried) or not np.any(r_buried):
                return 0.0, np.zeros(count)
            distance, _ = cKDTree(r.coor_xyz[r_buried]).query(q.coor_xyz[q_buried])
            areas = q.area[q_buried] * np.exp(-weight * distance ** 2)
            per_atom = np.bincount(q.atom_idx[q_buried] - offset,
                                   weights=areas, minlength=count)
            return float(per_atom.sum()), per_atom

    heavy = ~np.isin(atoms.element, ["H", "D"])
    indices_a, indices_b = np.flatnonzero(a & heavy), np.flatnonzero(b & heavy)
    reported_indices = indices_a if surface_side == "a" else indices_b
    radii = []
    for side in (indices_a, indices_b):
        values = np.asarray(module.get_radii_from_names(
            atoms.res_name[side].tolist(), atoms.atom_name[side].tolist()), dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("py-contact-ms has no valid SC radius for a selected atom")
        radii.append(values)
    calc = Surface()
    calc.settings.weight = weight
    # Nonoverlapping expanded spheres cannot create a buried molecular dot.
    # The upstream calculator otherwise crashes on some empty surface arrays.
    max_reach = float(np.max(radii[0]) + np.max(radii[1]) + 2 * calc.settings.rp)
    distance, _ = cKDTree(atoms.coord[indices_a]).query(atoms.coord[indices_b])
    empty_interface = bool(np.all(distance > max_reach))
    if empty_interface:
        value, per_atom = 0.0, np.zeros(len(reported_indices))
    else:
        calc.add_binder_and_target(atoms.coord[indices_a], radii[0],
                                   atoms.coord[indices_b], radii[1])
        value, per_atom = calc.CalcLoaded()
        if surface_side == "a":
            value, per_atom = calc.calc_contact_molecular_surface(target_side=False)
    if value < 0 or len(per_atom) != len(reported_indices) or not np.isfinite(per_atom).all():
        raise ValueError("Invalid reported-side CMS result")
    return _result(
        value, "contact_molecular_surface/py-contact-ms", backend="py-contact-ms",
        units="angstrom^2", definition="sum of reported-side buried dot area * exp(-weight * gap^2)",
        radii="py-contact-ms bundled SC radii", distance_weight=weight,
        surface_side=surface_side, target_side=surface_side.upper(),
        empty_interface=empty_interface,
        target_atom_indices=reported_indices.tolist(), per_target_atom=per_atom.tolist(),
        surface_scope="selected canonical amino-acid atoms; unselected atoms excluded",
        neighbor_search="nearest buried dots via scipy.cKDTree; atom neighbors still use dense distances",
        surface_pair_chunk_size=128,
    )


_CANONICAL = frozenset("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
_ACCEPTORS = {
    "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}, "ASN": {"OD1"},
    "GLN": {"OE1"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "CYS": {"SG"}, "MET": {"SD"}, "HIS": {"ND1", "NE2"},
}
_DONORS = {
    "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"}, "GLN": {"NE2"},
    "LYS": {"NZ"}, "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "TRP": {"NE1"},
}
_prepared: OrderedDict[tuple, Any] = OrderedDict()
_prepared_lock = threading.RLock()


def _fingerprint(atoms):
    h = hashlib.sha256()
    h.update(np.asarray(atoms.coord, dtype="<f8").tobytes())
    for category in sorted(atoms.get_annotation_categories()):
        h.update(category.encode() + b"\0")
        h.update(repr(np.asarray(atoms.get_annotation(category)).tolist()).encode())
    if atoms.bonds is not None:
        h.update(atoms.bonds.as_array().tobytes())
    return h.hexdigest()


def _hydrogenated(atoms, hydrogen_policy, ph):
    if hydrogen_policy not in {"add", "require"}:
        raise ValueError("hydrogen_policy must be 'add' or 'require'")
    if not math.isfinite(float(ph)) or not 0 <= float(ph) <= 14:
        raise ValueError("ph must be finite and between 0 and 14")
    unsupported = set(atoms.res_name) - _CANONICAL - {"HOH", "WAT"}
    if unsupported:
        raise BackendUnavailable(
            "Independent geometric H-bond typing supports canonical amino acids and water; "
            f"unsupported residues: {sorted(unsupported)}. Select the Rosetta backend."
        )
    key = (_fingerprint(atoms), hydrogen_policy, float(ph))
    with _prepared_lock:
        if key in _prepared:
            _prepared.move_to_end(key)
            return _prepared[key]
    struc = _module("biotite.structure", "biotite")
    hydrogen = np.isin(atoms.element, ["H", "D"])
    work = atoms.copy()
    if work.bonds is None:
        work.bonds = struc.connect_via_residue_names(work)
    if hydrogen_policy == "require":
        if not hydrogen.any():
            raise ValueError("hydrogen_policy='require' needs explicit hydrogen atoms")
        original_indices = np.arange(len(work))
    else:
        hydride = _module("hydride")
        original = np.flatnonzero(~hydrogen)
        work = work[~hydrogen]
        charges = hydride.estimate_amino_acid_charges(work, ph=float(ph))
        # Hydride identifies the first residue as a free terminus by order.
        # A head-to-tail connection makes that nitrogen an amide instead.
        for left, right, _ in work.bonds.as_array():
            for ni, ci in ((left, right), (right, left)):
                if (work.atom_name[ni] == "N" and work.atom_name[ci] == "C"
                        and (work.chain_id[ni], work.res_id[ni], work.ins_code[ni])
                        != (work.chain_id[ci], work.res_id[ci], work.ins_code[ci])):
                    charges[ni] = 0
        work.set_annotation("charge", charges)
        # Incomplete H placement must not silently create a plausible zero count.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                work, original_mask = hydride.add_hydrogen(work)
            except Warning as exc:
                raise ValueError(f"Hydrogen preparation is incomplete: {exc}") from exc
        original_indices = np.full(len(work), -1, dtype=int)
        original_indices[np.flatnonzero(original_mask)] = original
    _validate_atoms(work)
    donors, acceptors = _polar_types(work)
    if hydrogen_policy == "require":
        missing = []
        charges = work.charge if "charge" in work.get_annotation_categories() else np.zeros(len(work))
        for i, (res, name) in enumerate(zip(work.res_name, work.atom_name)):
            expected = (name == "N" and res in _CANONICAL and res != "PRO") or name in _DONORS.get(res, set())
            if expected and charges[i] >= 0 and not donors[i]:
                missing.append(f"{work.chain_id[i]}:{work.res_id[i]}:{name}")
        if missing:
            raise ValueError("Explicit hydrogen coverage is incomplete at donors: " + ", ".join(missing[:8]))
    result = (work, original_indices, donors, acceptors)
    with _prepared_lock:
        _prepared[key] = result
        _prepared.move_to_end(key)
        while len(_prepared) > 8:
            _prepared.popitem(last=False)
    return result


def _polar_types(atoms):
    """Canonical residue chemistry narrows Biotite's element-only acceptors."""
    hydrogen = np.isin(atoms.element, ["H", "D"])
    attached_h = np.zeros(len(atoms), dtype=bool)
    for i, j, _ in atoms.bonds.as_array():
        if hydrogen[j]:
            attached_h[i] = True
        if hydrogen[i]:
            attached_h[j] = True
    donor = attached_h & np.isin(atoms.element, ["N", "O", "S"])
    acceptor = np.zeros(len(atoms), dtype=bool)
    charge = (atoms.charge if "charge" in atoms.get_annotation_categories()
              else np.zeros(len(atoms)))
    for i, (res, name) in enumerate(zip(atoms.res_name, atoms.atom_name)):
        allowed = name in {"O", "OXT"} or name in _ACCEPTORS.get(res, set())
        if res in {"HOH", "WAT"} and atoms.element[i] == "O":
            allowed = True
        if res == "HIS" and name in {"ND1", "NE2"} and attached_h[i]:
            allowed = False
        carboxyl_oxygen = (name == "OXT" or (res == "ASP" and name in {"OD1", "OD2"})
                           or (res == "GLU" and name in {"OE1", "OE2"}))
        if carboxyl_oxygen and attached_h[i]:
            allowed = False
        acceptor[i] = allowed and charge[i] <= 0 and not hydrogen[i]
    return donor, acceptor


def _hbonds(atoms, donors, acceptors, distance, angle):
    if not math.isfinite(float(distance)) or float(distance) <= 0:
        raise ValueError("cutoff_distance must be finite and positive")
    if not math.isfinite(float(angle)) or not 0 < float(angle) <= 180:
        raise ValueError("cutoff_angle must be in (0, 180]")
    if not donors.any() or not acceptors.any():
        return np.empty((0, 3), dtype=int)
    struc = _module("biotite.structure", "biotite")
    return np.asarray(struc.hbond(
        atoms, donors, acceptors, selection1_type="donor",
        cutoff_dist=float(distance), cutoff_angle=float(angle),
    ), dtype=int).reshape(-1, 3)


def _remap(mask, original):
    result = np.zeros(len(original), dtype=bool)
    real = original >= 0
    result[real] = mask[original[real]]
    return result


def _hydrogen_details(policy, ph, distance, angle):
    return dict(hydrogen_policy=policy, ph=float(ph), cutoff_distance=float(distance),
                cutoff_angle=float(angle), hydrogen_preparation=(
                    "Hydride fragment placement; no heavy-atom movement or H relaxation"
                    if policy == "add" else "input hydrogen coordinates"),
                hydride_version=_version("hydride") if policy == "add" else None,
                chemistry="canonical amino-acid donor/acceptor typing; no Rosetta energies")


def hbond_count(atoms, mask_a, mask_b, *, cutoff_distance=2.5, cutoff_angle=120.0,
                hydrogen_policy="add", ph=7.0):
    """Count unique cross-selection donor-H-acceptor geometric observations."""
    a, b = _sides(atoms, mask_a, mask_b)
    work, original, donors, acceptors = _hydrogenated(atoms, hydrogen_policy, ph)
    a, b = _remap(a, original), _remap(b, original)
    triplets = _hbonds(work, donors, acceptors, cutoff_distance, cutoff_angle)
    inter = ((a[triplets[:, 0]] & b[triplets[:, 2]]) |
             (b[triplets[:, 0]] & a[triplets[:, 2]]))
    triplets = np.unique(triplets[inter], axis=0)
    return _result(
        len(triplets), "baker_hubbard_hbond/biotite", backend="biotite", units="count",
        definition="unique cross-side donor-H-acceptor triples satisfying geometric cutoffs",
        **_hydrogen_details(hydrogen_policy, ph, cutoff_distance, cutoff_angle),
    )


def interface_buns(
    atoms, mask_a, mask_b, *, group_a=None, group_b=None, cutoff_distance=2.5,
    cutoff_angle=120.0, hydrogen_policy="add", ph=7.0, sasa_probe_radius=1.1,
    buried_sasa_cutoff=0.2, sasa_point_number=1000,
):
    """Nonnegative bound-minus-separated geometric polar-heavy-atom BUNS.

    Use the full input as the bound environment; separate full chain groups and
    report only selected atoms. Same-side/spectator H bonds satisfy polar atoms.
    """
    a, b = _sides(atoms, mask_a, mask_b)
    ga = np.isin(atoms.chain_id, np.unique(atoms.chain_id[a])) if group_a is None else group_a
    gb = np.isin(atoms.chain_id, np.unique(atoms.chain_id[b])) if group_b is None else group_b
    ga, gb = _sides(atoms, ga, gb)
    if np.any(a & ~ga) or np.any(b & ~gb):
        raise ValueError("BUNS report selections must be contained in their full groups")
    probe, cutoff = float(sasa_probe_radius), float(buried_sasa_cutoff)
    if not math.isfinite(probe) or probe <= 0 or not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError("SASA probe radius must be positive and burial cutoff nonnegative")
    if isinstance(sasa_point_number, bool) or not isinstance(sasa_point_number, (int, np.integer)) or sasa_point_number < 20:
        raise ValueError("sasa_point_number must be an integer >= 20")
    work, original, donors, acceptors = _hydrogenated(atoms, hydrogen_policy, ph)
    a, b = _remap(a, original), _remap(b, original)
    # Generated H inherit their residue's chain and are separated with the group.
    ga_chains, gb_chains = np.unique(atoms.chain_id[ga]), np.unique(atoms.chain_id[gb])
    if np.intersect1d(ga_chains, gb_chains).size:
        raise ValueError("BUNS separated groups must be whole distinct chains")
    ga_work, gb_work = np.isin(work.chain_id, ga_chains), np.isin(work.chain_id, gb_chains)
    struc = _module("biotite.structure", "biotite")

    def count(keep, report):
        current = work[keep]
        d, ac = donors[keep], acceptors[keep]
        hb = _hbonds(current, d, ac, cutoff_distance, cutoff_angle)
        satisfied = np.zeros(len(current), dtype=bool)
        satisfied[hb[:, 0]] = True
        satisfied[hb[:, 2]] = True
        area = struc.sasa(current, probe_radius=probe,
                          point_number=int(sasa_point_number), vdw_radii="ProtOr")
        considered = report[keep] & (d | ac)
        if not np.isfinite(area[considered]).all():
            raise ValueError("SASA is undefined for a selected polar heavy atom")
        return int(np.count_nonzero(considered & (area <= cutoff) & ~satisfied))

    bound = count(np.ones(len(work), dtype=bool), a | b)
    unbound = count(~gb_work, a) + count(~ga_work, b)
    return _result(
        max(0, bound - unbound), "geometric_interface_buns/biotite", backend="biotite",
        units="count", definition="max(0, bound-unsatisfied-heavy-atoms minus separated-unsatisfied-heavy-atoms)",
        bound_buns=bound, separated_buns=unbound, sasa_probe_radius=probe,
        separation_policy="remove the opposite complete chain group for each report side; retain spectators",
        bound_environment="complete input structure; count only selected polar heavy atoms",
        buried_sasa_cutoff=cutoff, sasa_point_number=int(sasa_point_number),
        radii="Biotite ProtOr", **_hydrogen_details(hydrogen_policy, ph, cutoff_distance, cutoff_angle),
    )
