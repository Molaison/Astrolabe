"""Independent molecular surface and chemically typed geometric polar scores.

SC uses sc-rs, CMS uses py-contact-ms, and H-bonds use Baker-Hubbard geometry
after RDKit donor/acceptor typing. These backends do not reproduce Rosetta's
energy-based H-bond or DAlphaBall burial definitions.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import math
from functools import lru_cache
from pathlib import Path

import numpy as np


def require_backend(module_name, distribution):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Independent scoring requires {distribution}. Install "
            "Astrolabe[scoring], or explicitly select backend='rosetta'."
        ) from exc


def validate_backend(backend, independent):
    if backend not in (independent, "rosetta"):
        raise ValueError(
            f"backend must be {independent!r} or 'rosetta', got {backend!r}"
        )
    return backend



def validate_rosetta_polar_options(hydrogens, hydrogen_ph, hbond_distance, hbond_angle,
                                   sasa_probe_radius=1.4, buried_sasa_cutoff=1.0,
                                   sasa_points=1000):
    """Prevent geometry controls silently changing nothing on the energy backend."""
    actual = (hydrogens, hydrogen_ph, hbond_distance, hbond_angle,
              sasa_probe_radius, buried_sasa_cutoff, sasa_points)
    if actual != ("add", 7.0, 2.5, 120.0, 1.4, 1.0, 1000):
        raise ValueError("Hydrogen, geometry, and atom-SASA controls require backend='biotite'")

def _finite_positive(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (number < 0 if zero else number <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return number


def structure_context(eval_result, name):
    """Cache one immutable input snapshot per oligomer, keyed by file contents."""
    paths = getattr(eval_result, "complex_paths", None) or {}
    if name not in paths:
        raise ValueError(
            f"Independent scoring needs a structure path for {name!r}; "
            "use backend='rosetta' for a live Pose without an artifact."
        )
    path = Path(paths[name])
    content = path.read_bytes()
    digest = hashlib.sha256(content).digest()
    cache = getattr(eval_result, "_independent_structure_cache", None)
    if cache is None:
        cache = {}
        setattr(eval_result, "_independent_structure_cache", cache)
    previous = cache.get(name)
    if previous is not None and previous["digest"] == digest:
        return previous
    require_backend("biotite.structure", "biotite")
    from biotite.structure.io import pdb, pdbx

    text = io.StringIO(content.decode("utf-8"))
    if path.suffix.lower() in (".pdb", ".ent"):
        atoms = pdb.PDBFile.read(text).get_structure(
            model=1, altloc="occupancy", include_bonds=True, extra_fields=["charge"]
        )
    elif path.suffix.lower() in (".cif", ".mmcif"):
        atoms = pdbx.get_structure(
            pdbx.CIFFile.read(text), model=1, altloc="occupancy",
            use_author_fields=True, include_bonds=True, extra_fields=["charge"],
        )
    else:
        raise ValueError(f"Independent scoring expects PDB or mmCIF, got {path}")
    if not len(atoms) or not np.isfinite(atoms.coord).all():
        raise ValueError(f"Empty or nonfinite atomic coordinates in {path}")
    context = {"digest": digest, "atoms": atoms, "polar": {}, "surfaces": {}}
    cache[name] = context
    return context


def _surface_atoms(atoms, mask):
    selected = atoms[np.asarray(mask, dtype=bool) & ~np.isin(atoms.element, ["H", "D"])]
    if not len(selected):
        raise ValueError("Molecular surface selection contains no heavy atoms")
    if not np.isfinite(selected.coord).all():
        raise ValueError("Molecular surface coordinates must be finite")
    return selected


def independent_sc(atoms, mask_a, mask_b):
    """Lawrence-Colman SC with sc-rs radii, including negative geometric values."""
    backend = require_backend("protein_interface", "protein-interface==0.1.3")
    a, b = _surface_atoms(atoms, mask_a), _surface_atoms(atoms, mask_b)
    result = backend.compute_sc(
        coords_a=np.asarray(a.coord, dtype=float),
        atom_names_a=a.atom_name.tolist(), residue_names_a=a.res_name.tolist(),
        coords_b=np.asarray(b.coord, dtype=float),
        atom_names_b=b.atom_name.tolist(), residue_names_b=b.res_name.tolist(),
        parallel=True, use_spatial_index=True,
    )
    value = float(result.sc)
    if not math.isfinite(value) or not -1.000001 <= value <= 1.000001:
        raise ValueError(f"sc-rs returned invalid shape complementarity {value!r}")
    return value


def independent_cms(atoms, mask_a, mask_b, *, distance_weight=0.5, surface_cache=None):
    """Target-side distance-weighted molecular surface area in Angstrom squared.

    The pinned upstream implementation supplies the entire Connolly surface.
    Only its final all-pairs nearest-dot reduction is replaced by an exact
    KD-tree query; the published area*exp(-weight*d^2) definition is unchanged.
    Prepared area/distance arrays can be reused across weights and loss aliases.
    """
    backend = require_backend("py_contact_ms", "py-contact-ms==0.1.1")
    from py_contact_ms._core import MolecularSurfaceCalculator
    from scipy.spatial import cKDTree

    weight = _finite_positive(distance_weight, "distance_weight", zero=True)
    a, b = _surface_atoms(atoms, mask_a), _surface_atoms(atoms, mask_b)
    radii = [
        np.asarray(backend.get_radii_from_names(x.res_name.tolist(), x.atom_name.tolist()))
        for x in (a, b)
    ]
    for side, values in zip((a, b), radii):
        bad = ~np.isfinite(values) | (values <= 0)
        if np.any(bad):
            unsupported = sorted(set(zip(side.res_name[bad], side.atom_name[bad])))
            raise ValueError(f"py-contact-ms has no valid molecular-surface radii for {unsupported}")
    key = hashlib.sha256(
        a.coord.tobytes() + radii[0].tobytes() + b.coord.tobytes() + radii[1].tobytes()
    ).digest()
    cache = {} if surface_cache is None else surface_cache
    if key not in cache:
        calc = MolecularSurfaceCalculator()
        # Upstream's empty convex-queue path has a float-index bug. Prove no
        # interface attention atoms exist, using the upstream distance setting.
        nearest, _ = cKDTree(b.coord).query(a.coord, distance_upper_bound=calc.settings.sep)
        if not np.isfinite(nearest).any():
            cache[key] = (np.empty(0), np.empty(0), np.empty(0, dtype=int), len(b))
        else:
            calc.add_binder_and_target(a.coord, radii[0], b.coord, radii[1])
            # This is CalcLoaded()'s exact preparation, before its dense cdist.
            calc.run.atoms.finalize()
            calc.assign_attention_numbers(calc.run.atoms)
            calc.generate_molecular_surfaces()
            for dots in calc.run.dots:
                dots.finalize()
            query, reference = calc.run.dots[1], calc.run.dots[0]
            q = query.buried.astype(bool)
            r = reference.buried.astype(bool)
            if not np.any(q) or not np.any(r):
                cache[key] = (np.empty(0), np.empty(0), np.empty(0, dtype=int), len(b))
            else:
                distances, _ = cKDTree(reference.coor_xyz[r]).query(query.coor_xyz[q])
                cache[key] = (
                    query.area[q].copy(), distances * distances,
                    query.atom_idx[q].astype(int) - len(a), len(b),
                )
    area, distance_squared, indices, target_count = cache[key]
    # Preserve the upstream per-atom summation order as well as its definition.
    per_atom = np.bincount(indices, weights=area * np.exp(-weight * distance_squared),
                           minlength=target_count)
    value = float(per_atom.sum())
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"py-contact-ms returned invalid surface area {value!r}")
    return value


@lru_cache(maxsize=1)
def _feature_factory():
    require_backend("rdkit", "rdkit")
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    return ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))



def _typed_molecule(atoms):
    from biotite.structure import BondType
    from rdkit import Chem

    types = {
        BondType.SINGLE: Chem.BondType.SINGLE, BondType.DOUBLE: Chem.BondType.DOUBLE,
        BondType.TRIPLE: Chem.BondType.TRIPLE, BondType.QUADRUPLE: Chem.BondType.QUADRUPLE,
        BondType.AROMATIC_SINGLE: Chem.BondType.AROMATIC,
        BondType.AROMATIC_DOUBLE: Chem.BondType.AROMATIC,
        BondType.AROMATIC_TRIPLE: Chem.BondType.AROMATIC,
    }
    molecule = Chem.RWMol()
    for element, charge in zip(atoms.element, atoms.charge):
        atom = Chem.Atom("H" if element == "D" else str(element))
        atom.SetFormalCharge(int(charge))
        atom.SetNoImplicit(True)
        if element == "D":
            atom.SetIsotope(2)
        molecule.AddAtom(atom)
    for first, second, kind in atoms.bonds.as_array():
        try:
            bond_type = types[BondType(int(kind))]
        except (KeyError, ValueError) as exc:
            raise ValueError("H-bond typing requires known covalent bond orders") from exc
        molecule.AddBond(int(first), int(second), bond_type)
    molecule = molecule.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule

def polar_context(context, *, hydrogens="add", hydrogen_ph=7.0):
    """Place ideal-geometry H once; existing-H mode preserves supplied H geometry.

    This does not minimize coordinates or model local protonation equilibria.
    """
    if hydrogens not in ("add", "existing"):
        raise ValueError("hydrogens must be 'add' or 'existing'")
    ph = float(hydrogen_ph)
    if isinstance(hydrogen_ph, (bool, np.bool_)) or not math.isfinite(ph) or not 0 <= ph <= 14:
        raise ValueError("hydrogen_ph must be a finite number between 0 and 14")
    key = (hydrogens, ph)
    if key in context["polar"]:
        return context["polar"][key]
    import biotite.structure as struc
    from rdkit import Chem

    atoms = context["atoms"].copy()
    if atoms.bonds is None:
        raise ValueError("Geometric H-bond scoring requires a chemically typed BondList")
    if hydrogens == "add":
        hydride = require_backend("hydride", "hydride==1.2.3")
        heavy = atoms[~np.isin(atoms.element, ["H", "D"])]
        charges = heavy.charge.copy()
        for residue in struc.residue_iter(heavy):
            res_name = str(residue.res_name[0])
            try:
                reference = struc.info.residue(res_name)
            except KeyError as exc:
                raise ValueError(
                    f"Cannot infer chemical bonds/charges for {res_name!r}; provide "
                    "complete explicit hydrogens and bonds or select backend='rosetta'"
                ) from exc
            ccd_names = struc.info.get_from_ccd("chem_comp_atom", res_name, "atom_id").as_array()
            leaving = struc.info.get_from_ccd(
                "chem_comp_atom", res_name, "pdbx_leaving_atom_flag"
            ).as_array()
            leaving_names = set(ccd_names[leaving == "Y"])
            required = set(reference.atom_name[~np.isin(reference.element, ["H", "D"])]) - leaving_names
            missing = required - set(residue.atom_name)
            if missing:
                raise ValueError(
                    f"Incomplete heavy-atom chemistry for {res_name}: missing {sorted(missing)}; "
                    "hydrogen placement cannot reconstruct missing heavy atoms"
                )
            lookup = dict(zip(reference.atom_name, reference.charge))
            selected = (
                (heavy.chain_id == residue.chain_id[0])
                & (heavy.res_id == residue.res_id[0])
                & (heavy.ins_code == residue.ins_code[0])
            )
            for index in np.flatnonzero(selected):
                if heavy.atom_name[index] not in lookup:
                    raise ValueError(f"Unknown CCD atom {res_name}:{heavy.atom_name[index]}")
                charges[index] = lookup[heavy.atom_name[index]]
        canonical = struc.filter_canonical_amino_acids(heavy)
        charges[canonical] = hydride.estimate_amino_acid_charges(heavy, ph=ph)[canonical]
        heavy.set_annotation("charge", charges)
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            atoms, original_mask = hydride.add_hydrogen(heavy)
        if caught:
            raise ValueError("Hydrogen preparation was incomplete: " + "; ".join(str(x.message) for x in caught))
        if not np.array_equal(atoms.coord[original_mask], heavy.coord):
            raise RuntimeError("Hydrogen preparation changed heavy-atom coordinates")
    elif not np.any(atoms.element == "H"):
        raise ValueError("hydrogens='existing' requires explicit hydrogen atoms; no H-bonds can be inferred from an H-free structure")
    if not np.isfinite(atoms.coord).all():
        raise ValueError("Hydrogen preparation produced nonfinite coordinates")
    try:
        mol = _typed_molecule(atoms)
        if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
            raise ValueError(
                "Incomplete explicit hydrogens or unsupported radical chemistry; "
                "provide a complete closed-shell structure or use hydrogens='add'"
            )
    except Exception as exc:
        raise ValueError(f"Cannot type donor/acceptor chemistry from the input chemical graph: {exc}") from exc
    donors = np.zeros(len(atoms), dtype=bool)
    acceptors = np.zeros(len(atoms), dtype=bool)
    for feature in _feature_factory().GetFeaturesForMol(mol):
        if feature.GetFamily() == "Donor":
            donors[list(feature.GetAtomIds())] = True
        elif feature.GetFamily() == "Acceptor":
            acceptors[list(feature.GetAtomIds())] = True
    prepared = {"atoms": atoms, "donors": donors, "acceptors": acceptors, "hbonds": {}, "sasa": {}}
    context["polar"][key] = prepared
    return prepared


def hydrogen_bonds(prepared, *, hbond_distance=2.5, hbond_angle=120.0):
    """Unique D-H-A observations satisfying geometry and chemical atom typing."""
    distance = _finite_positive(hbond_distance, "hbond_distance")
    angle = _finite_positive(hbond_angle, "hbond_angle")
    if angle >= 180:
        raise ValueError("hbond_angle must be less than 180 degrees")
    key = (distance, angle)
    if key not in prepared["hbonds"]:
        import biotite.structure as struc

        triplets = struc.hbond(prepared["atoms"], cutoff_dist=distance, cutoff_angle=angle)
        valid = prepared["donors"][triplets[:, 0]] & prepared["acceptors"][triplets[:, 2]]
        prepared["hbonds"][key] = np.unique(triplets[valid], axis=0)
    return prepared["hbonds"][key]


def crossgroup_hbond_count(prepared, mask_a, mask_b, **geometry):
    triplets = hydrogen_bonds(prepared, **geometry)
    d, a = triplets[:, 0], triplets[:, 2]
    return float(np.count_nonzero((mask_a[d] & mask_b[a]) | (mask_b[d] & mask_a[a])))


def _unsatisfied_count(prepared, selection, *, sasa_probe_radius, buried_sasa_cutoff, sasa_points, geometry):
    import biotite.structure as struc

    probe = _finite_positive(sasa_probe_radius, "sasa_probe_radius")
    cutoff = _finite_positive(buried_sasa_cutoff, "buried_sasa_cutoff", zero=True)
    if isinstance(sasa_points, (bool, np.bool_)) or not isinstance(sasa_points, (int, np.integer)) or sasa_points < 20:
        raise ValueError("sasa_points must be an integer of at least 20")
    atoms = prepared["atoms"]
    key = (probe, int(sasa_points))
    polar = prepared["donors"] | prepared["acceptors"]
    if key not in prepared["sasa"]:
        prepared["sasa"][key] = struc.sasa(
            atoms, probe_radius=probe, atom_filter=polar, point_number=int(sasa_points),
            vdw_radii="Single", ignore_ions=False,
        )
    values = prepared["sasa"][key]
    if not np.isfinite(values[polar]).all():
        raise ValueError("SASA calculation failed for a polar atom")
    satisfied = np.zeros(len(atoms), dtype=bool)
    triplets = hydrogen_bonds(prepared, **geometry)
    satisfied[triplets[:, 0]] = True
    satisfied[triplets[:, 2]] = True
    return int(np.count_nonzero(selection & polar & (values <= cutoff) & ~satisfied))


def buried_unsatisfied_count(prepared, selection, *, sasa_probe_radius=1.4,
                             buried_sasa_cutoff=1.0, sasa_points=1000,
                             hbond_distance=2.5, hbond_angle=120.0):
    return _unsatisfied_count(
        prepared, selection, sasa_probe_radius=sasa_probe_radius,
        buried_sasa_cutoff=buried_sasa_cutoff, sasa_points=sasa_points,
        geometry={"hbond_distance": hbond_distance, "hbond_angle": hbond_angle},
    )


def interface_buns(prepared, mask_a, mask_b, chains_a, chains_b, **options):
    """Positive bound-minus-unbound unsatisfied heavy-atom count."""
    atoms = prepared["atoms"]
    group_a = np.isin(atoms.chain_id, chains_a)
    group_b = np.isin(atoms.chain_id, chains_b)
    if np.any(group_a & group_b) or not np.all(group_a | group_b):
        raise ValueError("Interface BUNS chain groups must be disjoint and cover the full structure")
    selected = mask_a | mask_b
    bound = buried_unsatisfied_count(prepared, selected, **options)
    unbound = 0
    for group in (group_a, group_b):
        isolated = {
            "atoms": atoms[group], "donors": prepared["donors"][group],
            "acceptors": prepared["acceptors"][group], "hbonds": {}, "sasa": {},
        }
        unbound += buried_unsatisfied_count(isolated, selected[group], **options)
    return float(max(0, bound - unbound))



def _cached_surface(context, metric, atoms, mask_a, mask_b, options=None):
    # Two differently weighted/normalized loss aliases can share the exact
    # surface result without sharing scientific settings or stale coordinates.
    options = options or {}
    key = (metric, mask_a.tobytes(), mask_b.tobytes(), tuple(sorted(options.items())))
    cache = context.setdefault("surfaces", {})
    if key not in cache:
        if metric == "sc":
            cache[key] = independent_sc(atoms, mask_a, mask_b)
        else:
            cache[key] = independent_cms(
                atoms, mask_a, mask_b, surface_cache=context.setdefault("cms_surfaces", {}),
                **options,
            )
    return cache[key]

def interface_scores(eval_result, metric, protomer_a=None, protomer_b=None,
                     included_protomers=None, *, selector_a=None, selector_b=None,
                     protomer_a_residues=None, protomer_b_residues=None,
                     hydrogens="add", hydrogen_ph=7.0, **options):
    from astrolabe.losses.selection import loss_atom_mask, resolve_loss_chain_groups

    scores = {}
    for name in eval_result.names:
        chains_a, chains_b = resolve_loss_chain_groups(
            name, protomer_a, protomer_b, included_protomers
        )
        context = structure_context(eval_result, name)
        prepared = None
        if metric in ("hbonds", "buns"):
            prepared = polar_context(context, hydrogens=hydrogens, hydrogen_ph=hydrogen_ph)
        atoms = context["atoms"] if prepared is None else prepared["atoms"]
        mask_a = loss_atom_mask(
            eval_result, name, atoms, chains_a, selector=selector_a, residues=protomer_a_residues
        )
        mask_b = loss_atom_mask(
            eval_result, name, atoms, chains_b, selector=selector_b, residues=protomer_b_residues
        )
        if np.any(mask_a & mask_b):
            raise ValueError("Interface atom selections must be disjoint")
        if metric == "sc":
            scores[name] = _cached_surface(context, "sc", atoms, mask_a, mask_b)
        elif metric == "cms":
            scores[name] = _cached_surface(context, "cms", atoms, mask_a, mask_b, options)
        elif metric == "hbonds":
            scores[name] = crossgroup_hbond_count(prepared, mask_a, mask_b, **options)
        elif metric == "buns":
            scores[name] = interface_buns(prepared, mask_a, mask_b, chains_a, chains_b, **options)
        else:
            raise ValueError(f"Unknown independent interface metric {metric!r}")
    return scores


def ligand_scores(eval_result, metric, ligand_targets=None, *, selector_a=None,
                  selector_b=None, neighbor_distance=8.0, hydrogens="add",
                  hydrogen_ph=7.0, **options):
    from astrolabe.losses.interaction import _resolve_ligand_chains, _protein_chains_for_oligo
    from astrolabe.losses.selection import loss_atom_mask
    import biotite.structure as struc

    scores = {}
    for name in eval_result.names:
        ligands, _ = _resolve_ligand_chains(name, ligand_targets)
        proteins = _protein_chains_for_oligo(name)
        if not ligands or not proteins:
            raise ValueError(f"Ligand scoring requires ligand and protein chains in {name!r}")
        context = structure_context(eval_result, name)
        prepared = None if metric == "sc" else polar_context(
            context, hydrogens=hydrogens, hydrogen_ph=hydrogen_ph
        )
        atoms = context["atoms"] if prepared is None else prepared["atoms"]
        protein_mask = loss_atom_mask(eval_result, name, atoms, proteins, selector=selector_b)
        values = []
        for _protomer, chain in ligands:
            ligand_mask = loss_atom_mask(eval_result, name, atoms, [chain], selector=selector_a)
            if metric == "sc":
                for protein in proteins:
                    partner = protein_mask & (atoms.chain_id == protein)
                    values.append(_cached_surface(context, "sc", atoms, partner, ligand_mask))
            elif metric == "hbonds":
                values.append(crossgroup_hbond_count(prepared, ligand_mask, protein_mask, **options))
            elif metric == "buns":
                from scipy.spatial import cKDTree
                distance = _finite_positive(neighbor_distance, "neighbor_distance")
                heavy = ~np.isin(atoms.element, ["H", "D"])
                if not np.any(ligand_mask & heavy):
                    raise ValueError("Ligand BUNS selection contains no heavy atoms")
                neighbors = cKDTree(atoms.coord[ligand_mask & heavy]).query_ball_point(
                    atoms.coord, distance, return_length=True
                ) > 0
                local = np.zeros(len(atoms), dtype=bool)
                starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
                for start, stop in zip(starts[:-1], starts[1:]):
                    if np.any(neighbors[start:stop] & heavy[start:stop]):
                        local[start:stop] = True
                selected = ligand_mask | (protein_mask & local)
                values.append(float(buried_unsatisfied_count(prepared, selected, **options)))
            else:
                raise ValueError(f"Unknown independent ligand metric {metric!r}")
        scores[name] = float(np.mean(values))
    return scores
