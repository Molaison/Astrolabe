"""Functional-atom motifs for explicitly variable, unindexed reconstruction.

Only native insertion annotations establish residue lineage. Newly generated
scaffold residues have no positional ancestry in their parent.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .rfd4_types import ConditionPlan, HBondCondition, ResidueRef


# Small chemical frames that retain the direction of a donor/acceptor. Planar
# amides and aromatic systems need their functional plane, not a whole residue.
_FRAMES = {
    "ASN": ("CG", "OD1", "ND2"),
    "GLN": ("CD", "OE1", "NE2"),
    "ASP": ("CG", "OD1", "OD2"),
    "GLU": ("CD", "OE1", "OE2"),
    "ARG": ("NE", "CZ", "NH1", "NH2"),
    "LYS": ("CD", "CE", "NZ"),
    "SER": ("CA", "CB", "OG"),
    "THR": ("CA", "CB", "OG1", "CG2"),
    "CYS": ("CA", "CB", "SG"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "TRP": ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}


def reconstruction_bounds(config):
    if not isinstance(config, dict) or set(config) != {"length_min", "length_max"}:
        raise ValueError("reconstruction requires exactly length_min and length_max")
    lower, upper = config["length_min"], config["length_max"]
    if type(lower) is not int or type(upper) is not int or lower < 2 or upper < lower:
        raise ValueError("reconstruction length bounds must be inclusive positive integers")
    return lower, upper


def functional_atoms(residue_name, atom):
    if atom == "N":
        return ("N", "CA", "C")
    if atom == "O":
        return ("CA", "C", "O")
    frame = _FRAMES.get(residue_name, ())
    if atom not in frame:
        raise ValueError(f"No functional geometry definition for {residue_name}:{atom}")
    return frame


def source_atom_names(path):
    """Physical label-chain and residue ordinal, matching the adapter contract."""
    from Bio.PDB import PDBParser
    path = Path(path)
    if path.suffix.lower() in {".cif", ".mmcif"}:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        rows = MMCIF2Dict(str(path))
        positions, result = {}, {}
        for chain, seq, auth, name in zip(
            rows["_atom_site.label_asym_id"], rows["_atom_site.label_seq_id"],
            rows["_atom_site.auth_seq_id"], rows["_atom_site.label_atom_id"], strict=True,
        ):
            keys = positions.setdefault(chain, {})
            key = (seq, auth)
            if key not in keys:
                keys[key] = len(keys) + 1
            result.setdefault(ResidueRef(chain, keys[key]), set()).add(name)
        return result
    structure = PDBParser(QUIET=True).get_structure("source", path)
    return {ResidueRef(chain.id, i): {a.name for a in residue}
            for chain in structure[0] for i, residue in enumerate(chain, 1)}


def read_motif_lineage(path):
    """Read only hash-verified native provenance beside a retained structure."""
    path = Path(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.name.endswith(".generated.cif"):
        stem = path.name.removesuffix(".generated.cif")
        index_path = path.with_name(stem + ".rfd4.generated.json")
        if not index_path.is_file():
            raise ValueError("Published generated input is missing its provenance index")
        index = json.loads(index_path.read_text())
        metadata_path = path.with_name(stem + ".backend.json")
        metadata_raw = metadata_path.read_bytes()
        if (index["files"][path.name]["sha256"] != sha
                or index["files"][metadata_path.name]["sha256"]
                != hashlib.sha256(metadata_raw).hexdigest()):
            raise ValueError("Generated motif lineage provenance digest mismatch")
        metadata = json.loads(metadata_raw)
    elif path.name == "generated.cif" and path.with_name("manifest.json").is_file():
        metadata = json.loads(path.with_name("manifest.json").read_text())
    else:
        return {}
    if metadata["checksums"]["generated.cif"] != sha:
        raise ValueError("Native motif lineage structure digest mismatch")
    reconstruction = metadata["request"]["plan"].get("reconstruction")
    if reconstruction is None:
        return {}
    requested = {m["id"]: m for m in reconstruction["motifs"]}
    mapping = metadata["validation"].get("motif_mapping", [])
    if len(mapping) != len(requested) or {m["id"] for m in mapping} != set(requested):
        raise ValueError("Native motif lineage mapping is incomplete")
    lineage = {}
    for item in mapping:
        original = requested[item["id"]]
        if item["source"] != original["source"] or set(item["atoms"]) != set(original["atoms"]):
            raise ValueError("Native motif lineage mapping disagrees with its request")
        site = ResidueRef(**item["output"])
        if site in lineage:
            raise ValueError("Native motif lineage mapping is not injective")
        lineage[site] = item["id"]
    return lineage


def _geometry_qualified(hbond):
    donor, hydrogen, acceptor = hbond.donor_xyz, hbond.hydrogen_xyz, hbond.acceptor_xyz
    distance = math.dist(donor, acceptor)
    a = [d - h for d, h in zip(donor, hydrogen)]
    b = [v - h for v, h in zip(acceptor, hydrogen)]
    denominator = math.sqrt(sum(v*v for v in a) * sum(v*v for v in b))
    if denominator <= 0:
        return False
    angle = math.degrees(math.acos(max(-1, min(1, sum(x*y for x, y in zip(a, b))/denominator))))
    return distance <= 3.5 and angle >= 120


def plan_reconstruction(path, chains, feedback, *, config, policy=None,
                        excluded_amino_acids=(), failed_hbond_attempts=None):
    lower, upper = reconstruction_bounds(config)
    policy = dict(policy or {})
    if policy.get("energy_top_k", 0) or policy.get("repair_segment_length") is not None:
        raise ValueError("Unindexed reconstruction retains functional HBs, not fixed packing residues/windows")
    cutoff = policy.get("hbond_retention_cutoff", -0.5)
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff) or cutoff > 0:
        raise ValueError("HB retention cutoff must be finite and nonpositive")
    if policy.get("freeze_hbonds", True) is not True:
        raise ValueError("Functional reconstruction requires existing good HB protection")
    maximum = policy.get("max_hbond_conditions", 1)
    if type(maximum) is not int or maximum not in (0, 1):
        raise ValueError("Functional reconstruction supports at most one unsatisfied ligand goal")
    binder = [c for c in chains if c.kind == "protein"]
    ligands = [c for c in chains if c.kind == "ligand"]
    if len(binder) != 1 or not ligands or len(binder) + len(ligands) != len(chains):
        raise ValueError("Functional reconstruction requires one binder and fixed single-CCD ligands")
    binder = binder[0]
    if feedback is None or tuple(feedback.binder_chains) != (binder.chain,):
        raise ValueError("Functional reconstruction requires current accepted interface feedback")
    atom_names = source_atom_names(path)
    lineage = read_motif_lineage(path)
    source_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    residues = {r.residue: r for r in feedback.residues}
    polar = {a.atom: a for a in feedback.polar_atoms}
    retained, protected_hbonds, omissions = {}, [], []
    for hb in feedback.interfacial_hbonds:
        if hb.energy >= cutoff or not _geometry_qualified(hb):
            continue
        if any(a not in polar or not polar[a].source_present or a.atom == "OXT"
               for a in (hb.donor, hb.acceptor)):
            omissions.append({"hbond": asdict(hb), "reason": "heavy atom absent from source/native representation"})
            continue
        atom = hb.donor if hb.donor.chain == binder.chain else hb.acceptor
        site = ResidueRef(atom.chain, atom.position)
        frame = functional_atoms(residues[site].residue_name, atom.atom)
        if not set(frame) <= atom_names.get(site, set()):
            raise ValueError(f"Source motif {site} lacks functional atoms {frame}")
        retained.setdefault(site, set()).update(frame)
        protected_hbonds.append(asdict(hb))
    motifs = [
        {"id": lineage.get(site) or hashlib.sha256(
            f"{source_sha}:{site.chain}:{site.position}".encode()).hexdigest(),
         "source": asdict(site), "atoms": sorted(names)}
        for site, names in sorted(retained.items())
    ]
    if len(motifs) >= lower:
        raise ValueError("Length bounds must leave space for connecting scaffold residues")
    failed = dict(failed_hbond_attempts or {})
    goals = sorted(
        (a for a in feedback.polar_atoms if a.atom.chain in {c.chain for c in ligands}
         and a.source_present and not a.satisfied and a.atom.atom != "OXT"),
        key=lambda a: (failed.get(f"{a.atom.chain}:{a.atom.position}:{a.atom.atom}", 0),
                       not a.buried, a.sasa, a.atom),
    )
    conditions = tuple(HBondCondition(a.atom, a.role) for a in goals[:maximum])
    report = {
        "mode": "variable_unindexed_functional_geometry",
        "source_sha256": source_sha,
        "length_bounds": [lower, upper],
        "protected_hbonds": protected_hbonds,
        "motifs": motifs,
        "retention_cutoff": cutoff,
        "retention_geometry": {"max_donor_acceptor_angstrom": 3.5, "min_DHA_angle_degrees": 120},
        "omissions": omissions,
        "no_motif_behavior": "ligand-conditioned de novo scaffold; no positional ancestry",
        "coordinate_condition": "soft native C_CRD on functional atoms only",
        "sequence_condition": "motif residue identity only; C_IDX false with independent groups",
        "hbond_conditions": [asdict(c) for c in conditions],
        "goal_rule": "one unsatisfied ligand atom in its own chemical role; failed attempts then buried/SASA/identity",
        "failed_hbond_attempts": failed,
        "ancestry": "only native mapped motif residues; no ordinal correspondence for new scaffold",
    }
    return ConditionPlan(
        released=tuple(ResidueRef(binder.chain, i) for i in range(1, len(binder.sequence)+1)),
        fixed_coordinates=tuple(ResidueRef(c.chain, 1) for c in ligands),
        hbonds=conditions, excluded_amino_acids=tuple(excluded_amino_acids),
        feedback_json=json.dumps(report, sort_keys=True, allow_nan=False),
        reconstruction={"chain": binder.chain, "length_min": lower, "length_max": upper, "motifs": motifs},
    )
