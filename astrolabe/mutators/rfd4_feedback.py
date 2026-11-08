"""Atom-resolved interface observations and pure, budgeted RFD4 conditions.

Distances define the interface neighborhood, never hydrogen bonds. SASA and
selection defaults are protocol choices, not experimentally optimized cutoffs.
The signed EnergyGraph proxy excludes long-range and nonpair energy terms.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping

from .rfd4_types import AtomRef, ChainSpec, ConditionPlan, HBondCondition, ResidueRef


@dataclass(frozen=True)
class HBondObservation:
    donor: AtomRef
    hydrogen: AtomRef
    acceptor: AtomRef
    donor_xyz: tuple[float, float, float]
    hydrogen_xyz: tuple[float, float, float]
    acceptor_xyz: tuple[float, float, float]
    energy: float
    weight: float
    interchain: bool
    crossgroup: bool
    qualifying: bool


@dataclass(frozen=True)
class PolarAtomObservation:
    atom: AtomRef
    residue_name: str
    atom_type: str
    role: str
    xyz: tuple[float, float, float]
    sasa: float
    nearest_other_group_heavy_atom_distance: float
    buried: bool
    interface: bool
    satisfied: bool
    qualifying_hbond_count: int
    source_present: bool = True


@dataclass(frozen=True)
class ResidueObservation:
    residue: ResidueRef
    residue_name: str
    pose_index: int
    author_number: int
    insertion_code: str
    short_range_interface_pair_energy: float
    interface_hbond_count: int
    unsatisfied_interface_polar_count: int
    heavy_atom_coordinates: tuple[tuple[float, float, float], ...] = ()
    nearest_other_group_heavy_atom_distance: float | None = None


@dataclass(frozen=True)
class PairEnergyObservation:
    binder: ResidueRef
    target: ResidueRef
    short_range_interface_pair_energy: float
    weighted_terms: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class InterfaceFeedback:
    binder_chains: tuple[str, ...]
    target_chains: tuple[str, ...]
    residues: tuple[ResidueObservation, ...]
    hbonds: tuple[HBondObservation, ...]
    polar_atoms: tuple[PolarAtomObservation, ...]
    pair_energies: tuple[PairEnergyObservation, ...]
    shape_complementarity: float
    provenance: tuple[tuple[str, str], ...]
    interface_empty: bool = False

    @property
    def interfacial_hbonds(self) -> tuple[HBondObservation, ...]:
        return tuple(h for h in self.hbonds if h.crossgroup and h.qualifying)

    @property
    def interfacial_hbond_count(self) -> int:
        """Count qualifying crossgroup donor-H/acceptor observations once each."""
        return len(self.interfacial_hbonds)

    @property
    def interface_polar_count(self) -> int:
        return sum(a.interface for a in self.polar_atoms)

    @property
    def satisfied_interface_polar_count(self) -> int:
        return sum(a.interface and a.satisfied for a in self.polar_atoms)

    @property
    def interface_polar_satisfaction(self) -> float | None:
        """Fraction with any qualifying HB, not valence-complete saturation."""
        total = self.interface_polar_count
        return self.satisfied_interface_polar_count / total if total else None

    @property
    def short_range_interface_pair_energy(self) -> float:
        return sum(p.short_range_interface_pair_energy for p in self.pair_energies)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["provenance"] = dict(self.provenance)
        result["metrics"] = {
            "interface_polar_count": self.interface_polar_count,
            "satisfied_interface_polar_count": self.satisfied_interface_polar_count,
            "interface_polar_satisfaction": self.interface_polar_satisfaction,
            "interfacial_hbond_count": self.interfacial_hbond_count,
            "short_range_interface_pair_energy": self.short_range_interface_pair_energy,
        }
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)


def _groups(binder_chains, target_chains, available):
    binder = (binder_chains,) if isinstance(binder_chains, str) else tuple(binder_chains)
    target = (target_chains,) if isinstance(target_chains, str) else tuple(target_chains)
    if (not binder or not target or len(set(binder)) != len(binder)
            or len(set(target)) != len(target) or set(binder) & set(target)):
        raise ValueError("Binder and target must be nonempty, distinct physical chain groups")
    if set(binder) | set(target) != set(available):
        raise ValueError("Explicit binder/target groups must partition all physical chains")
    return binder, target


def _xyz(residue, atom):
    point = residue.xyz(atom)
    return float(point.x), float(point.y), float(point.z)


def _hbond_donor_atom(residue, hydrogen):
    """Resolve a polar H's base to its physical atom (e.g. N-terminal Pro NV→N)."""
    donor = residue.atom_base(hydrogen)
    if residue.atom_type(donor).is_virtual():
        shadowed = residue.type().atom_being_shadowed(donor)
        if shadowed:
            return shadowed
    return donor


def analyze_interface(
    structure_path, binder_chains, target_chains, *, extra_residue_params=None, **options,
) -> InterfaceFeedback:
    """Analyze a fresh accepted structure using physical, one-based chain ordinals.

    Binders are canonical proteins; each target is a canonical protein or one
    noncovalent, nonpolymeric ligand. Rosetta-added hydrogens are expected;
    silently reconstructed or discarded source heavy atoms are not.

    Reuse the exact .params generated by astrolabe.utils.extra_residues via
    extra_residue_params=[...], or pre-register them with PyRosetta -extra_res_fa.
    By default, also load the ordinary Astrolabe out_dir/raw_model_results/*.params.
    This function never parameterizes a ligand or substitutes its chemistry.
    """
    from Bio.PDB import MMCIFIO, MMCIFParser, PDBIO, PDBParser
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from Bio.Data.PDBData import protein_letters_3to1
    from io import StringIO
    import pyrosetta
    from astrolabe.core import global_config
    from astrolabe.utils.mmcif_fallback import _generate_nonstandard_residue_set

    path = Path(structure_path)
    is_mmcif = path.suffix.lower() in {".cif", ".mmcif"}
    chain_provenance = ()
    if is_mmcif:
        record = {key: value for key, value in MMCIF2Dict(str(path)).items()
                  if key == "data_" or key.startswith("_atom_site.")}
        # Native de novo scaffolds may omit B factors. Biopython requires this
        # metadata column; its neutral import default is not a confidence score.
        record.setdefault("_atom_site.B_iso_or_equiv",
                          ["0.0"] * len(record["_atom_site.label_atom_id"]))
        author_ids = {}
        for label, author in zip(record["_atom_site.label_asym_id"], record["_atom_site.auth_asym_id"], strict=True):
            author_ids.setdefault(label, set()).add(author)
        # Older Biopython cannot select label chains. Normalize the author
        # field before parsing and keep the original author mapping separately.
        record["_atom_site.auth_asym_id"] = list(record["_atom_site.label_asym_id"])
        writer = MMCIFIO()
        writer.set_dict(record)
        normalized_text = StringIO()
        writer.save(normalized_text)
        normalized_text = normalized_text.getvalue()
        models = list(MMCIFParser(QUIET=True).get_structure("feedback", StringIO(normalized_text)))
        chain_provenance = (
            ("source_chain_namespace", "mmCIF _atom_site.label_asym_id"),
            ("physical_chain_author_ids", json.dumps(
                {label: sorted(authors) for label, authors in author_ids.items()}, sort_keys=True,
            )),
            ("rosetta_chain_normalization", "physical label-chain PDB import; source atom coordinates restored after import; original author residue-number fields preserved"),
        )
    else:
        models = list(PDBParser(QUIET=True, PERMISSIVE=False).get_structure("feedback", str(path)))
    if len(models) != 1:
        raise ValueError("Interface feedback requires exactly one structure model")
    source = {}
    for chain in models[0]:
        residues = list(chain)
        if not residues or len(chain.id) != 1 or not chain.id.strip():
            raise ValueError("Rosetta feedback requires nonempty single-character physical chain IDs")
        if any(residue.is_disordered() for residue in residues):
            raise ValueError("Feedback does not support alternate conformers")
        if any(not math.isfinite(float(value)) for residue in residues
               for atom in residue for value in atom.coord):
            raise ValueError("Feedback requires finite source atom coordinates")
        source[chain.id] = residues
    binder, target = _groups(binder_chains, target_chains, source)
    ligand_chains = []
    for cid, residues in source.items():
        if all(residue.id[0] == " " and residue.resname in protein_letters_3to1
               for residue in residues):
            continue
        if (cid not in target or len(residues) != 1
                or not residues[0].id[0].startswith("H_")):
            raise ValueError("Feedback requires canonical protein binders and protein or single nonpolymeric ligand targets")
        ligand_chains.append(cid)
    if not pyrosetta.rosetta.basic.was_init_called():
        pyrosetta.init("-mute all")
    if extra_residue_params is None:
        try:
            extra_residue_params = sorted(Path(global_config.out_dir).glob("raw_model_results/*.params"))
        except RuntimeError:
            extra_residue_params = ()
    if isinstance(extra_residue_params, (str, Path)):
        extra_residue_params = (extra_residue_params,)
    params = tuple(str(param) for param in extra_residue_params)
    try:
        if is_mmcif or params:
            pose = pyrosetta.Pose()
            _generate_nonstandard_residue_set(pyrosetta, pose, params)
            if is_mmcif:
                writer = PDBIO()
                writer.set_structure(models[0])
                pdb_text = StringIO()
                writer.save(pdb_text)
                pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose, pdb_text.getvalue())
            else:
                pyrosetta.pose_from_file(pose, str(path))
        else:
            pose = pyrosetta.pose_from_file(str(path))
    except Exception as exc:
        if not ligand_chains:
            raise
        raise ValueError(
            f"Could not load target ligand chemistry for {ligand_chains}; register the exact "
            "Astrolabe .params with extra_residue_params or PyRosetta -extra_res_fa"
        ) from exc
    loaded_chains = [pose.pdb_info().chain(pose.chain_begin(ci))
                     for ci in range(1, pose.num_chains() + 1)]
    if len(loaded_chains) != len(source) or set(loaded_chains) != set(source):
        raise ValueError("Rosetta import dropped or changed physical chains (including target ligand); check exact extra-res .params")
    # Import must preserve all heavy-atom identities and coordinates. OXT can
    # be supplied by Rosetta's terminal patch when absent from the source.
    for ci in range(1, pose.num_chains() + 1):
        begin, end = pose.chain_begin(ci), pose.chain_end(ci)
        chain_id = pose.pdb_info().chain(begin)
        original = source.get(chain_id)
        if original is None or len(original) != end - begin + 1:
            raise ValueError("Rosetta import changed physical chains or residue ordinals")
        for ordinal, index in enumerate(range(begin, end + 1)):
            residue = pose.residue(index)
            raw = original[ordinal]
            if residue.name3() != raw.resname:
                raise ValueError("Rosetta import changed residue chemistry")
            heavy_names = {residue.atom_name(atom).strip()
                           for atom in range(1, residue.nheavyatoms() + 1)
                           if not residue.atom_type(atom).is_virtual()}
            for raw_atom in raw:
                if raw_atom.element not in {"H", "D"} and raw_atom.name not in heavy_names:
                    raise ValueError(f"Rosetta import dropped source heavy atom at {chain_id}:{ordinal + 1}:{raw_atom.name}")
            for atom in range(1, residue.nheavyatoms() + 1):
                if residue.atom_type(atom).is_virtual():
                    continue
                name = residue.atom_name(atom).strip()
                if residue.is_protein() and name == "OXT" and name not in raw:
                    continue
                if name not in raw or any(abs(a - float(b)) > 0.002 for a, b in zip(_xyz(residue, atom), raw[name].coord)):
                    raise ValueError(f"Missing or altered heavy atom at {chain_id}:{ordinal + 1}:{name}")
            # Restore full source precision after Rosetta's PDB import rounding,
            # including ligand coordinates (without rebuilding missing atoms).
            if is_mmcif or chain_id in ligand_chains:
                for atom in range(1, residue.natoms() + 1):
                    name = residue.atom_name(atom).strip()
                    if name in raw:
                        pose.set_xyz(
                            pyrosetta.rosetta.core.id.AtomID(atom, index),
                            pyrosetta.rosetta.numeric.xyzVector_double_t(*map(float, raw[name].coord)),
                        )
    feedback = analyze_pose_interface(pose, binder_chains, target_chains, **options)
    polar_atoms = tuple(replace(atom, source_present=(
        atom.atom.atom in source[atom.atom.chain][atom.atom.position - 1]
    )) for atom in feedback.polar_atoms)
    return replace(feedback, polar_atoms=polar_atoms, provenance=feedback.provenance + chain_provenance + (
        ("structure_sha256", hashlib.sha256(path.read_bytes()).hexdigest()),
        ("extra_residue_params", json.dumps([
            {"name": Path(param).name, "sha256": hashlib.sha256(Path(param).read_bytes()).hexdigest()}
            for param in params
        ], sort_keys=True)),
        ("terminal_OXT", "Rosetta terminal patch may add missing OXT; flagged source_present=False and excluded from retention/conditions"),
    ))


def analyze_pose_interface(
    pose, binder_chains, target_chains, *, hbond_energy_cutoff=0.0,
    interface_distance=5.0, buried_sasa_cutoff=1.0, sasa_probe_radius=1.4,
) -> InterfaceFeedback:
    """Shared analyzer for scoring adapters. Clone the caller's pose before scoring.

    A buried polar atom has bound-complex SASA <= buried_sasa_cutoff in A^2.
    Interface membership uses the nearest other-group heavy atom, <= the
    distance cutoff in A. Satisfaction requires any HBondSet bond with energy
    strictly below hbond_energy_cutoff, including same-chain bonds.
    """
    import numpy as np
    import pyrosetta
    from scipy.spatial import cKDTree

    thresholds = dict(hbond_energy_cutoff=hbond_energy_cutoff, interface_distance=interface_distance,
                      buried_sasa_cutoff=buried_sasa_cutoff, sasa_probe_radius=sasa_probe_radius)
    if any(not math.isfinite(float(v)) for v in thresholds.values()):
        raise ValueError("Interface thresholds must be finite")
    if interface_distance <= 0 or buried_sasa_cutoff < 0 or sasa_probe_radius <= 0:
        raise ValueError("Invalid distance, SASA cutoff or probe radius")
    pose = pose.clone()
    info = pose.pdb_info()
    if info is None or not pose.is_fullatom():
        raise ValueError("Feedback requires a full-atom pose with physical chain identities")
    refs, chains = {}, []
    chain_ranges = []
    for ci in range(1, pose.num_chains() + 1):
        begin, end = pose.chain_begin(ci), pose.chain_end(ci)
        cid = info.chain(begin)
        if not cid.strip() or cid in chains:
            raise ValueError("Physical chain IDs must be unique and nonblank")
        chains.append(cid)
        chain_ranges.append((cid, begin, end))
    binder, target = _groups(binder_chains, target_chains, chains)
    for cid, begin, end in chain_ranges:
        for ordinal, index in enumerate(range(begin, end + 1), 1):
            residue = pose.residue(index)
            chemistry = residue.type()
            protein = residue.is_protein() and chemistry.is_canonical_aa()
            ligand = (cid in target and begin == end and not residue.is_protein() and chemistry.is_ligand()
                      and not chemistry.is_polymer() and not chemistry.is_water()
                      and not chemistry.is_metal() and not chemistry.is_RNA()
                      and not chemistry.is_DNA() and residue.n_current_residue_connections() == 0)
            if (not (protein or ligand) or info.chain(index) != cid
                    or any(info.alt_loc(index, atom).strip() for atom in range(1, residue.natoms() + 1))):
                raise ValueError("Feedback requires canonical protein binders and consistently labeled protein or single noncovalent nonpolymeric ligand targets without alternate conformers (no water, metal or nucleic acid)")
            refs[index] = ResidueRef(cid, ordinal)
    binder_indices = [i for i, ref in refs.items() if ref.chain in binder]
    target_indices = [i for i, ref in refs.items() if ref.chain in target]
    r = pyrosetta.rosetta
    scorefxn = pyrosetta.create_score_function("ref2015")
    energy_options = r.core.scoring.methods.EnergyMethodOptions(scorefxn.energy_method_options())
    energy_options.hbond_options().decompose_bb_hb_into_pair_energies(True)
    scorefxn.set_energy_method_options(energy_options)
    scorefxn(pose)
    weights = scorefxn.weights()
    graph = pose.energies().energy_graph()
    score_types = tuple(graph.active_2b_score_types())
    energies = Counter({ref: 0.0 for ref in refs.values()})
    pairs = []
    for i in binder_indices:
        for j in target_indices:
            edge = graph.find_edge(i, j)
            if edge is None:
                continue
            terms = tuple((str(term).split(".")[-1], float(edge[term]) * float(weights[term]))
                          for term in score_types if edge[term] != 0 and weights[term] != 0)
            energy = sum(value for _, value in terms)
            if not math.isfinite(energy):
                raise ValueError("Nonfinite short-range pair energy")
            energies[refs[i]] += energy
            energies[refs[j]] += energy
            if terms:
                pairs.append(PairEnergyObservation(refs[i], refs[j], energy, terms))

    def atom_ref(index, atom):
        ref = refs[index]
        return AtomRef(ref.chain, ref.position, pose.residue(index).atom_name(atom).strip())

    hbond_options = r.core.scoring.hbonds.HBondOptions(energy_options.hbond_options())
    hbond_set = r.core.scoring.hbonds.HBondSet(hbond_options)
    r.core.scoring.hbonds.fill_hbond_set(pose, False, hbond_set)
    hbonds, satisfied, hbond_counts = [], Counter(), Counter()
    for index in range(1, hbond_set.nhbonds() + 1):
        hb = hbond_set.hbond(index)
        di, ai, hydrogen, acceptor = hb.don_res(), hb.acc_res(), hb.don_hatm(), hb.acc_atm()
        donor = _hbond_donor_atom(pose.residue(di), hydrogen)
        donor_ref, acceptor_ref = atom_ref(di, donor), atom_ref(ai, acceptor)
        crossgroup = (refs[di].chain in binder) != (refs[ai].chain in binder)
        energy = float(hb.energy())
        if not math.isfinite(energy):
            raise ValueError("Nonfinite HBondSet energy")
        qualifying = energy < hbond_energy_cutoff
        hbonds.append(HBondObservation(
            donor_ref, atom_ref(di, hydrogen), acceptor_ref,
            _xyz(pose.residue(di), donor), _xyz(pose.residue(di), hydrogen),
            _xyz(pose.residue(ai), acceptor), energy, float(hb.weight()),
            refs[di].chain != refs[ai].chain, crossgroup, qualifying,
        ))
        if qualifying:
            satisfied[donor_ref] += 1
            satisfied[acceptor_ref] += 1
            if crossgroup:
                hbond_counts[refs[di]] += 1
                hbond_counts[refs[ai]] += 1

    sasa = r.core.scoring.sasa.SasaCalc()
    sasa.set_calculation_method(r.core.scoring.sasa.SasaMethodEnum.LeGrand)
    sasa.set_probe_radius(float(sasa_probe_radius))
    sasa.set_include_hydrogens_explicitly(True)
    sasa.calculate(pose)
    atom_sasa = sasa.get_atom_sasa()
    trees, group_coordinates = {}, {}
    for group in (binder, target):
        coordinates = [_xyz(pose.residue(i), atom) for i, ref in refs.items() if ref.chain in group
                       for atom in range(1, pose.residue(i).nheavyatoms() + 1)
                       if not pose.residue(i).atom_type(atom).is_virtual()]
        group_coordinates[group] = np.asarray(coordinates, dtype=float)
        trees[group] = cKDTree(group_coordinates[group])
    nearest_heavy_distance = float(np.min(trees[target].query(group_coordinates[binder])[0]))
    interface_empty = nearest_heavy_distance > interface_distance
    polar_atoms, unsatisfied = [], Counter()
    for i, ref in refs.items():
        residue = pose.residue(i)
        # Match HBondSet's residue-level chemistry, including protonation and
        # terminal/disulfide variants. AtomType.is_donor() alone marks Npro as
        # a donor even when an internal Pro nitrogen has no hydrogen to donate.
        donor_atoms = {_hbond_donor_atom(residue, h) for h in residue.Hpos_polar()}
        acceptor_atoms = set(residue.accpt_pos())
        other_tree = trees[target if ref.chain in binder else binder]
        for atom in range(1, residue.nheavyatoms() + 1):
            chemistry = residue.atom_type(atom)
            donor, acceptor = atom in donor_atoms, atom in acceptor_atoms
            if chemistry.is_virtual() or not (donor or acceptor):
                continue
            identity = atom_ref(i, atom)
            xyz = _xyz(residue, atom)
            distance = float(other_tree.query(xyz)[0])
            area = float(atom_sasa(r.core.id.AtomID(atom, i)))
            if not math.isfinite(area) or area < 0:
                raise ValueError("Invalid atomic SASA")
            interface, buried = distance <= interface_distance, area <= buried_sasa_cutoff
            count = satisfied[identity]
            if interface and buried and not count:
                unsatisfied[ref] += 1
            polar_atoms.append(PolarAtomObservation(
                identity, residue.name3(), chemistry.name(),
                "BOTH" if donor and acceptor else "DONOR" if donor else "ACCEPTOR",
                xyz, area, distance, buried, interface, bool(count), count,
            ))
    if interface_empty:
        sc = 0.0
        sc_method = "measured empty interface neighborhood; SC=0; geometry backend not invoked"
    else:
        calculator = r.core.scoring.sc.ShapeComplementarityCalculator()
        if not calculator.Init():
            raise ValueError("Shape complementarity initialization failed")
        for i, ref in refs.items():
            if not calculator.AddResidue(0 if ref.chain in binder else 1, pose.residue(i)):
                raise ValueError("Shape complementarity rejected a group residue")
        if not calculator.Calc():
            raise ValueError("Group shape complementarity calculation failed")
        sc = float(calculator.GetResults().sc)
        sc_method = "ShapeComplementarityCalculator; group molecules 0/1; default radii"
    if sc is None or not math.isfinite(sc) or sc < 0:
        raise ValueError("Invalid shape complementarity scalar")
    residue_coordinates = {
        ref: tuple(_xyz(pose.residue(i), atom)
                   for atom in range(1, pose.residue(i).nheavyatoms() + 1)
                   if not pose.residue(i).atom_type(atom).is_virtual())
        for i, ref in refs.items()
    }
    residues = tuple(ResidueObservation(
        ref, pose.residue(i).name3(), i, info.number(i), info.icode(i).strip(),
        energies[ref], hbond_counts[ref], unsatisfied[ref],
        residue_coordinates[ref],
        float(np.min(trees[target if ref.chain in binder else binder].query(residue_coordinates[ref])[0])),
    ) for i, ref in refs.items())
    provenance = {
        "pyrosetta_version": pyrosetta.version(), "scorefunction": "ref2015",
        "decompose_bb_hb_into_pair_energies": "true",
        "pair_energy_method": "direct weighted EnergyGraph active_2b_score_types; same short-range scope as residue_pair_energies_array; signed; excludes long-range/nonpair terms; not ddG",
        "hbond_method": "Rosetta HBondSet; donor=physical shadow-normalized atom_base(don_hatm); energy < cutoff",
        "polar_role_definition": "donor=physical shadow-normalized atom_base(Hpos_polar); acceptor=accpt_pos; current Rosetta residue type including variants; BOTH=both roles",
        "satisfaction_definition": "any qualifying HB at the polar heavy atom, including same-chain; not valence-complete saturation",
        "interface_definition": "nearest other-group heavy atom distance <= interface_distance; not HB definition",
        "nearest_crossgroup_heavy_atom_distance": str(nearest_heavy_distance),
        "burial_definition": "bound-complex atomic SASA <= buried_sasa_cutoff; not interface-induced delta SASA",
        "sasa_method": "SasaCalc; LeGrand; explicit hydrogens",
        "shape_method": sc_method,
        "residue_identity": "physical chain and one-based chain ordinal; author number is metadata only",
        "threshold_status": "configurable protocol choices; not validated optima",
        **{key: str(value) for key, value in thresholds.items()},
    }
    return InterfaceFeedback(binder, target, residues, tuple(hbonds), tuple(polar_atoms),
                             tuple(pairs), float(sc), tuple(sorted(provenance.items())), interface_empty)


def _plan_local_repair(
    universe, allowed, local_allowed, observations, preserved, feedback, budget,
    binder, target, excluded, seed, scores, radius, hbond_cutoff, energy_cutoff,
    top_k, max_conditions, freeze_hbonds, native_unrepresented,
    native_omission_reason, native_retention_omissions, failed_hbond_attempts,
    segment_length=None,
):
    """Pure geometry/contact/energy proxy selection; no local SC inference."""
    candidates = sorted(local_allowed - set(preserved))

    def goal_distance(ref, atom):
        return min(math.dist(xyz, atom.xyz) for xyz in observations[ref].heavy_atom_coordinates)

    atoms = sorted((a for a in feedback.polar_atoms
                    if a.atom.chain in target and a.source_present and not a.satisfied
                    and a.nearest_other_group_heavy_atom_distance <= radius),
                   key=lambda a: (
                       failed_hbond_attempts.get(f"{a.atom.chain}:{a.atom.position}:{a.atom.atom}", 0),
                       not a.buried, a.sasa, a.atom)) if feedback else []
    omissions = [{"atom": asdict(a.atom), "role": a.role, "reason": native_omission_reason}
                 for a in atoms if native_unrepresented(a.atom)]
    windows = []
    if segment_length is not None:
        # Flanks may leave the distance shell, but may never cross an excluded
        # site, chain boundary, or protected interaction. Keep chain indices.
        usable = {ref for ref in allowed - set(preserved)
                  if ref in observations and observations[ref].heavy_atom_coordinates}
        for first in sorted(usable):
            window = tuple(ResidueRef(first.chain, first.position + i)
                           for i in range(segment_length))
            if set(window) <= usable and set(window) & local_allowed:
                windows.append(window)
    goals = [a for a in atoms if not native_unrepresented(a.atom)
             and any(goal_distance(ref, a) <= radius for ref in
                     ({r for window in windows for r in window}
                      if segment_length is not None else candidates))]
    goal = goals[0] if goals and max_conditions else None
    selection_candidates = [ref for ref in candidates
                            if goal is None or goal_distance(ref, goal) <= radius]

    def rank(ref):
        obs = observations[ref]
        proxy_rank = (-obs.unsatisfied_interface_polar_count,
                      -obs.short_range_interface_pair_energy, ref)
        return (goal_distance(ref, goal), *proxy_rank) if goal else proxy_rank

    if segment_length is None:
        released = tuple(sorted(sorted(selection_candidates, key=rank)[:budget]))
    else:
        windows = [window for window in windows if goal is None
                   or min(goal_distance(ref, goal) for ref in window) <= radius]

        def window_rank(window):
            proxy = (-sum(observations[r].unsatisfied_interface_polar_count for r in window),
                     -sum(observations[r].short_range_interface_pair_energy for r in window),
                     window)
            return (min(goal_distance(r, goal) for r in window), *proxy) if goal else proxy

        released = min(windows, key=window_rank) if windows else ()
        selection_candidates = sorted({r for window in windows for r in window})
    released_set = set(released)
    fixed = tuple(ref for ref in universe if ref not in released_set)
    conditions = []
    if goal and any(goal_distance(ref, goal) <= radius for ref in released):
        if goal.role not in {"DONOR", "ACCEPTOR", "BOTH"}:
            raise ValueError("Unknown chemical role in interface feedback")
        conditions.append(HBondCondition(goal.atom, goal.role))

    reasons, local_details = [], []
    for ref in universe:
        obs = observations.get(ref)
        if ref in preserved:
            reason = sorted(set(preserved[ref]))
        elif ref in released_set:
            reason = (["released_as_contiguous_repair_segment"] if segment_length is not None else
                      ["released_by_goal_distance_unsat_pair_energy_identity" if goal else
                       "released_by_interface_unsat_pair_energy_identity"])
        elif ref.chain in target:
            reason = ["fixed_target"]
        elif ref not in allowed:
            reason = ["outside_eligible_mask"]
        elif not obs or not obs.heavy_atom_coordinates or obs.nearest_other_group_heavy_atom_distance is None:
            reason = ["missing_local_geometry"]
        elif ref not in local_allowed:
            reason = ["outside_interface_repair_radius"]
        elif goal and ref not in selection_candidates:
            reason = ["outside_selected_goal_radius"]
        else:
            reason = ["global_budget_limit"]
        reasons.append({"residue": asdict(ref), "reasons": reason})
        if ref.chain in binder:
            local_details.append({
                "residue": asdict(ref), "allowed": ref in allowed,
                "local_eligible": ref in local_allowed, "candidate": ref in candidates,
                "selection_candidate": ref in selection_candidates,
                "released": ref in released_set, "protected_reasons": [] if ref in released_set else reason,
                "nearest_target_heavy_atom_distance": obs.nearest_other_group_heavy_atom_distance if obs else None,
                "selected_goal_heavy_atom_distance": goal_distance(ref, goal)
                if goal and obs and obs.heavy_atom_coordinates else None,
                "unsatisfied_interface_polar_count": obs.unsatisfied_interface_polar_count if obs else None,
                "short_range_interface_pair_energy": obs.short_range_interface_pair_energy if obs else None,
            })
    report = {
        "feedback": feedback.to_dict() if feedback else None,
        "planning": {
            "budget": budget, "seed": seed, "binder_chains": binder, "target_chains": target,
            "policy": {"local_repair": True, "freeze_hbonds": freeze_hbonds,
                       "repair_radius": radius, "hbond_retention_cutoff": hbond_cutoff,
                       "energy_retention_cutoff": energy_cutoff, "energy_top_k": top_k,
                       "max_hbond_conditions": max_conditions},
            "effective_max_hbond_conditions": min(max_conditions, 1),
            "protocol_defaults_status": "configurable protocol choices; not validated optima",
            "eligible": [asdict(ref) for ref in sorted(allowed)],
            "local_eligible": [asdict(ref) for ref in sorted(local_allowed)],
            "local_candidates": [asdict(ref) for ref in candidates],
            "selection_candidates": [asdict(ref) for ref in selection_candidates],
            "selected_goal": asdict(goal) if goal else None,
            "failed_hbond_attempts": failed_hbond_attempts,
            "local_residues": local_details, "chosen_reasons": reasons,
            "local_mask_reason": "allowed binder residues with measured nearest target heavy distance <= repair_radius; when a goal is selected, also require goal heavy distance <= repair_radius before ranking and budget; freeze complement and protect whole residues",
            "rationale": "retain source-present qualifying crossgroup HB below hbond_retention_cutoff and top-k pair energies below energy_retention_cutoff; choose reachable unsatisfied target polar goal by buried first, SASA, atom identity; restrict selection to the selected goal radius, then release by goal distance, unsatisfied count, descending pair energy, stable identity, or the same proxy rank without distance if no goal; no random changes or local shape-complementarity claims",
            "site_scores": [{"residue": asdict(ref), "score": float(scores[ref])} for ref in candidates] if scores is not None else None,
            "site_scores_effect": "not used by deterministic local repair ranking",
            "hbond_conditions": [asdict(c) for c in conditions],
            "hbond_condition_reason": "at most one source-present target polar atom in its own role, never OXT, and only with a released binder heavy atom within repair_radius",
            "native_unrepresented_atoms": [
                {"atom": asdict(a.atom), "source_present": a.source_present, "reason": native_omission_reason}
                for a in feedback.polar_atoms if native_unrepresented(a.atom)
            ] if feedback else [],
            "native_unrepresented_atom_condition_omissions": omissions,
            "native_unrepresented_hbond_retention_omissions": native_retention_omissions,
        },
    }
    if segment_length is not None:
        planning = report["planning"]
        planning["policy"]["repair_segment_length"] = segment_length
        planning["repair_segment"] = {
            "length": segment_length,
            "residues": [asdict(ref) for ref in released],
            "feasible_windows": len(windows),
            "status": "selected" if released else "no_complete_eligible_segment",
            "outside_radius_flanks": [asdict(ref) for ref in released
                                     if ref not in local_allowed],
        }
        planning["local_mask_reason"] = (
            "one complete same-chain eligible contiguous segment, never crossing protected "
            "interactions; at least one member within repair_radius of the selected goal "
            "(or interface when no goal); connecting flanks may lie outside the radius")
        planning["rationale"] = (
            "retain good interactions; rank reachable unsatisfied goals by failed attempts, "
            "burial, SASA and atom identity; rank complete windows by nearest goal distance, "
            "summed unsatisfied count, descending summed pair energy and stable identity; "
            "no sparse fallback or partial-window truncation")
    return ConditionPlan(released, fixed, tuple(conditions), excluded,
                         json.dumps(report, sort_keys=True, allow_nan=False))


def plan_conditions(
    chains: tuple[ChainSpec, ...], eligible: tuple[ResidueRef, ...], budget: int, *,
    feedback: InterfaceFeedback | None = None, policy=None, site_scores=None,
    seed=0, excluded_amino_acids=(), failed_hbond_attempts=None,
) -> ConditionPlan:
    """Release at most the global budget before inference, without broadening masks.

    Policy defaults freeze_hbonds=True, energy_top_k=0, max_hbond_conditions=8
    are configurable protocol choices. Explicit chain groups are required when
    feedback is absent, including when the mutable mask is empty. Opt-in
    local_repair uses radius 6.0 A, HB retention < -0.5, pair-energy retention
    < -1.0, energy_top_k=2 and max_hbond_conditions=1 unless overridden.
    """
    policy = dict(policy or {})
    supported = {"binder_chains", "target_chains", "freeze_hbonds", "energy_top_k", "max_hbond_conditions",
                 "local_repair", "repair_radius", "hbond_retention_cutoff", "energy_retention_cutoff",
                 "repair_segment_length"}
    if set(policy) - supported:
        raise ValueError(f"Unknown feedback policy keys: {sorted(set(policy) - supported)}")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("Mutation budget must be a nonnegative integer")
    local_repair = policy.get("local_repair", False)
    if not isinstance(local_repair, bool):
        raise ValueError("local_repair must be boolean")
    segment_length = policy.get("repair_segment_length")
    if "repair_segment_length" in policy and (
        not local_repair or isinstance(segment_length, bool)
        or not isinstance(segment_length, int) or segment_length < 2 or segment_length > budget
    ):
        raise ValueError("repair_segment_length requires local_repair and an integer from 2 through budget")
    failed_hbond_attempts = dict(failed_hbond_attempts or {})
    if (failed_hbond_attempts and not local_repair) or any(
        not isinstance(key, str) or not re.fullmatch(r"[^:]+:[1-9][0-9]*:[^:]+", key)
        or isinstance(value, bool) or not isinstance(value, int) or value < 0
        for key, value in failed_hbond_attempts.items()
    ):
        raise ValueError("Failed HB attempts require local repair and nonnegative integer atom counts")
    radius = policy.get("repair_radius", 6.0)
    hbond_cutoff = policy.get("hbond_retention_cutoff", -0.5)
    energy_cutoff = policy.get("energy_retention_cutoff", -1.0)
    for name, value in (("repair_radius", radius), ("hbond_retention_cutoff", hbond_cutoff),
                        ("energy_retention_cutoff", energy_cutoff)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if radius <= 0:
        raise ValueError("repair_radius must be positive")
    if hbond_cutoff > 0 or energy_cutoff > 0:
        raise ValueError("Retention cutoffs must be nonpositive")
    top_k = policy.get("energy_top_k", 2 if local_repair else 0)
    max_conditions = policy.get("max_hbond_conditions", 1 if local_repair else 8)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (top_k, max_conditions)):
        raise ValueError("Retention and HBond condition limits must be nonnegative integers")
    freeze_hbonds = policy.get("freeze_hbonds", True)
    if not isinstance(freeze_hbonds, bool):
        raise ValueError("freeze_hbonds must be boolean")
    ids = [chain.chain for chain in chains]
    specs = {chain.chain: chain for chain in chains}
    if (len(ids) != len(set(ids)) or any(not cid.strip() or len(cid) != 1 for cid in ids)
            or any(c.kind not in {"protein", "ligand"} or not c.sequence
                   or (c.kind == "protein" and set(c.sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
                   or (c.kind == "ligand" and not re.fullmatch(r"[A-Z0-9]+", c.sequence))
                   for c in chains)):
        raise ValueError("Chains must have unique single-character identities and canonical protein sequences or one explicit ligand CCD identifier")
    binder = policy.get("binder_chains", feedback.binder_chains if feedback else None)
    target = policy.get("target_chains", feedback.target_chains if feedback else None)
    if binder is None or target is None:
        raise ValueError("Without feedback, policy must explicitly set binder_chains and target_chains")
    binder, target = _groups(binder, target, ids)
    if any(specs[cid].kind != "protein" for cid in binder):
        raise ValueError("Binder chains must be proteins; fixed ligands are target-only and never releasable")
    if feedback and (set(binder) != set(feedback.binder_chains) or set(target) != set(feedback.target_chains)):
        raise ValueError("Policy chain groups disagree with interface feedback")
    universe = tuple(ResidueRef(c.chain, i) for c in chains for i in range(1, c.residue_count + 1))
    all_sites, allowed = set(universe), set(eligible)
    if len(allowed) != len(eligible) or not allowed <= all_sites or any(r.chain not in binder for r in allowed):
        raise ValueError("Eligible sites must be unique binder residue ordinals within the input chains")
    excluded = tuple(sorted(set(excluded_amino_acids)))
    if any(aa not in tuple("ACDEFGHIKLMNPQRSTVWY") for aa in excluded) or len(excluded) == 20:
        raise ValueError("Excluded amino acids must leave at least one canonical choice")
    observations = {r.residue: r for r in feedback.residues} if feedback else {}
    if feedback:
        from Bio.SeqUtils import seq1

        if len(observations) != len(feedback.residues) or set(observations) != all_sites:
            raise ValueError("Feedback residue identities disagree with the planned chains")
        if any((obs.residue_name != specs[ref.chain].sequence if specs[ref.chain].kind == "ligand"
                else seq1(obs.residue_name) != specs[ref.chain].sequence[ref.position - 1])
               for ref, obs in observations.items()):
            raise ValueError("Feedback residue chemistry disagrees with the planned sequences")

    def native_unrepresented(atom):
        return (local_repair or specs[atom.chain].kind == "protein") and atom.atom == "OXT"

    local_allowed = set()
    if local_repair:
        for ref, obs in observations.items():
            distance = obs.nearest_other_group_heavy_atom_distance
            if (distance is not None and (not math.isfinite(distance) or distance < 0)
                    or any(len(xyz) != 3 or any(not math.isfinite(v) for v in xyz)
                           for xyz in obs.heavy_atom_coordinates)):
                raise ValueError("Local repair requires finite heavy atom geometry and nonnegative distances")
            if ref in allowed and obs.heavy_atom_coordinates and distance is not None and distance <= radius:
                local_allowed.add(ref)
    retention_allowed = local_allowed if local_repair and segment_length is None else allowed
    preserved = {}
    native_omission_reason = "OXT_not_represented_by_native_RFD4; original_chemistry_observations_retained"
    native_condition_omissions, native_retention_omissions = [], []
    if feedback and freeze_hbonds:
        absent_atoms = {a.atom for a in feedback.polar_atoms if not a.source_present}
        source_atoms = {a.atom for a in feedback.polar_atoms if a.source_present}
        for hb in feedback.interfacial_hbonds:
            if hb.energy >= (hbond_cutoff if local_repair else 0):
                continue
            unrepresented = [atom for atom in (hb.donor, hb.acceptor) if native_unrepresented(atom)]
            if unrepresented:
                native_retention_omissions.append({
                    "hbond": asdict(hb), "atoms": [asdict(atom) for atom in unrepresented],
                    "reason": native_omission_reason,
                })
                continue
            if hb.donor in absent_atoms or hb.acceptor in absent_atoms:
                continue
            if local_repair and (hb.donor not in source_atoms or hb.acceptor not in source_atoms):
                continue
            for atom in (hb.donor, hb.acceptor):
                ref = ResidueRef(atom.chain, atom.position)
                if ref in retention_allowed:
                    preserved.setdefault(ref, []).append("favorable_interfacial_hbond")
    if feedback and top_k:
        favorable = sorted((r for r in feedback.residues if r.residue in retention_allowed
                            and r.short_range_interface_pair_energy < (energy_cutoff if local_repair else 0)),
                           key=lambda r: (r.short_range_interface_pair_energy, r.residue))
        for residue in favorable[:top_k]:
            preserved.setdefault(residue.residue, []).append("favorable_short_range_pair_energy")
    candidates = sorted(retention_allowed - set(preserved))
    scores = dict(site_scores) if site_scores is not None else None
    if scores is not None and any(ref not in scores or not math.isfinite(float(scores[ref])) for ref in candidates):
        raise ValueError("site_scores must provide a finite score for every releasable eligible site")
    if local_repair:
        return _plan_local_repair(
            universe, allowed, local_allowed, observations, preserved, feedback, budget,
            binder, target, excluded, seed, scores, radius, hbond_cutoff, energy_cutoff,
            top_k, max_conditions, freeze_hbonds, native_unrepresented,
            native_omission_reason, native_retention_omissions, failed_hbond_attempts,
            segment_length=segment_length,
        )
    rng = random.Random(seed)
    ties = {ref: rng.random() for ref in candidates}

    def rank(ref):
        if scores is not None:
            return (-float(scores[ref]), ties[ref], ref)
        obs = observations.get(ref)
        return (-(obs.short_range_interface_pair_energy if obs else 0),
                -(obs.unsatisfied_interface_polar_count if obs else 0), ties[ref], ref)

    released = tuple(sorted(sorted(candidates, key=rank)[:budget]))
    fixed = tuple(ref for ref in universe if ref not in set(released))
    conditions = []
    if feedback:
        atoms = sorted((a for a in feedback.polar_atoms if a.atom.chain in target
                        and a.interface and a.buried and not a.satisfied and a.source_present),
                       key=lambda a: (a.sasa, a.atom))
        native_condition_omissions = [
            {"atom": asdict(a.atom), "role": a.role, "reason": native_omission_reason}
            for a in atoms if native_unrepresented(a.atom)
        ]
        atoms = [a for a in atoms if not native_unrepresented(a.atom)]
        for atom in atoms[:max_conditions]:
            if atom.role not in {"DONOR", "ACCEPTOR", "BOTH"}:
                raise ValueError("Unknown chemical role in interface feedback")
            conditions.append(HBondCondition(atom.atom, atom.role))
    if any(ResidueRef(c.atom.chain, c.atom.position) not in fixed for c in conditions):
        raise ValueError("HBond conditions must reference retained atom identities")
    reasons = []
    for ref in universe:
        reason = preserved.get(ref)
        if reason is None:
            reason = (["released_by_site_scores" if scores is not None else "released_by_pair_energy_unsat_seeded_rank"]
                      if ref in released else ["fixed_target" if ref.chain in target else
                                              "outside_eligible_mask" if ref not in allowed else "global_budget_limit"])
        reasons.append({"residue": asdict(ref), "reasons": sorted(set(reason))})
    report = {
        "feedback": feedback.to_dict() if feedback else None,
        "planning": {
            "budget": budget, "seed": seed, "binder_chains": binder, "target_chains": target,
            "policy": {"freeze_hbonds": freeze_hbonds, "energy_top_k": top_k,
                       "max_hbond_conditions": max_conditions},
            "protocol_defaults_status": "configurable protocol choices; not validated optima",
            "eligible": [asdict(r) for r in sorted(allowed)], "chosen_reasons": reasons,
            "site_scores": [{"residue": asdict(r), "score": float(scores[r])} for r in candidates] if scores is not None else None,
            "hbond_conditions": [asdict(c) for c in conditions],
            "hbond_condition_reason": "fixed target buried interface polar atom without any qualifying HB; selected atom's own role",
            "native_unrepresented_atoms": [
                {"atom": asdict(a.atom), "source_present": a.source_present, "reason": native_omission_reason}
                for a in feedback.polar_atoms if native_unrepresented(a.atom)
            ] if feedback else [],
            "native_unrepresented_atom_condition_omissions": native_condition_omissions,
            "native_unrepresented_hbond_retention_omissions": native_retention_omissions,
        },
    }
    return ConditionPlan(released, fixed, tuple(conditions), excluded,
                         json.dumps(report, sort_keys=True, allow_nan=False))
