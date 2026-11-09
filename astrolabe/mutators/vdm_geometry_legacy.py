#!/usr/bin/env python3
"""Natural vdM pair rigid-backbone matching and sequence proposal.

Library JSON: {"schema_version": 1, "pairs": [{"his": {"aa":"H",
"backbone":{"N":[...],"CA":[...],"C":[...]}, "sidechain":{atom: xyz}},
"partner": {...}, "source": {...}}]}. Coordinates are Angstrom, in one frame.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SeqUtils import seq1
from scipy.spatial import cKDTree

BACKBONE = ("N", "CA", "C")
MAINCHAIN = {"N", "CA", "C", "O", "OXT"}
RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}


def read_structure(path, chain):
    path = Path(path)
    parser = MMCIFParser(QUIET=True) if path.suffix.lower() in (".cif", ".mmcif") else PDBParser(QUIET=True)
    model = next(parser.get_structure("input", str(path)).get_models())
    if chain not in model:
        raise ValueError(f"chain {chain!r} missing from {path}")
    residues = [r for r in model[chain] if r.id[0] == " "]
    seq = ''.join(seq1(r.resname, custom_map={"UNK": "X"}) for r in residues)
    return model, residues, seq


def fit(mobile, target):
    """Least-squares proper (rotation determinant +1) fit, row-vector convention."""
    mobile, target = np.asarray(mobile, float), np.asarray(target, float)
    cm, ct = mobile.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile-cm).T @ (target-ct))
    d = np.eye(3)
    d[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ d @ vt
    translation = ct - cm @ rotation
    residual = mobile @ rotation + translation - target
    return rotation, translation, float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def _element(name):
    return name.strip().upper()[0]


def direct_pair_hbond(pair):
    """Require reciprocal vdM sidechain H-bond annotations, not X backbone O/N."""
    contacts = pair.get('source', {}).get('hbond_contacts', [])
    x = {row.get('name'): {v.strip() for v in (row.get('partners_hb') or '').split(',')}
         for row in contacts if row.get('chain') == 'X'}
    y = {row.get('name'): {v.strip() for v in (row.get('partners_hb') or '').split(',')}
         for row in contacts if row.get('chain') == 'Y'}
    partner = pair['partner']['aa']
    donor_h = {'K': {'HZ1','HZ2','HZ3'}, 'R': {'HE','HH11','HH12','HH21','HH22'},
               'H': {'HD1','HE2'}}[partner]
    for nitrogen in ('ND1','NE2'):
        for hydrogen in donor_h:
            if hydrogen in y.get(nitrogen, set()) and nitrogen in x.get(hydrogen, set()):
                return True
    if partner == 'H':
        for hydrogen in ('HD1','HE2'):
            for nitrogen in ('ND1','NE2'):
                if nitrogen in y.get(hydrogen, set()) and hydrogen in x.get(nitrogen, set()):
                    return True
    return False


def _environment(model, residues):
    target_ids = {id(residue): pos for pos, residue in enumerate(residues)}
    env = []
    for atom in model.get_atoms():
        name = atom.get_name().strip()
        element = (atom.element or _element(name)).upper()
        if element not in RADII:
            continue
        residue = atom.get_parent()
        env.append((np.asarray(atom.coord, float), element, target_ids.get(id(residue)), name))
    return env, cKDTree(np.array([a[0] for a in env]))


def _clash(environment, residues, pair_positions, pair, rotation, translation, scale):
    # Exclude only sidechains being replaced; preserve their backbones.
    env, tree = environment
    placed = []
    for position, unit in zip(pair_positions, (pair['his'], pair['partner'])):
        for name, coord in unit['sidechain'].items():
            element = _element(name)
            if element not in RADII or name in MAINCHAIN:
                continue
            placed.append((np.asarray(coord, float) @ rotation + translation, element, position, name))
    if not placed:
        raise ValueError('pair lacks heavy-atom sidechains')
    # Exclude intramolecular contacts with the same residue's own backbone,
    # which are covalent/local; evaluate cross-residue atoms and pair overlap.
    for xyz, element, position, name in placed:
        for index in tree.query_ball_point(xyz, scale * (RADII[element] + max(RADII.values()))):
            other, element2, target, other_name = env[index]
            if target == position:
                continue
            if target in pair_positions and other_name not in MAINCHAIN:
                continue
            distance = float(np.linalg.norm(xyz-other))
            if distance < scale * (RADII[element]+RADII[element2]):
                return {"atom": name, "distance": distance, "kind": "environment"}
    for i, (xyz, element, position, name) in enumerate(placed):
        for other, element2, pos2, name2 in placed[i+1:]:
            if position == pos2:
                continue
            distance = float(np.linalg.norm(xyz-other))
            if distance < scale * (RADII[element]+RADII[element2]):
                return {"atom": name, "other_atom": name2, "distance": distance, "kind": "pair"}
    return None


def _attachment(residues, positions, pair, rotation, translation):
    """Compare placed CA-CB length and N-CA-CB/C-CA-CB angles to source."""
    def angle(a, b):
        return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))))
    result = []
    for pos, unit in zip(positions, (pair['his'], pair['partner'])):
        if 'CB' not in unit['sidechain']:
            raise ValueError('pair lacks CB required for sidechain attachment')
        source_ca = np.asarray(unit['backbone']['CA'], float)
        source_cb = np.asarray(unit['sidechain']['CB'], float)
        placed_cb = source_cb @ rotation + translation
        target_ca = residues[pos]['CA'].coord
        source_vector, target_vector = source_cb-source_ca, placed_cb-target_ca
        length_delta = abs(float(np.linalg.norm(target_vector)-np.linalg.norm(source_vector)))
        angles = {}
        for a in ('N','C'):
            source_angle = angle(np.asarray(unit['backbone'][a])-source_ca, source_vector)
            target_angle = angle(residues[pos][a].coord-target_ca, target_vector)
            angles[a] = abs(target_angle-source_angle)
        result.append({'position_zero_based':pos,'ca_cb_length_delta_A':length_delta,'angle_delta_degrees':angles})
    return result


def propose_pair(*, structure_path, chain, sequence, mutable_positions, library_path,
                 reference_sequence, max_total_mutations, max_edits, parameters=None):
    params = dict(parameters or {})
    max_rmsd = float(params.get('max_backbone_rmsd', 1.0))
    clash_scale = float(params.get('clash_radius_scale', 0.75))
    max_bond_delta = float(params.get('max_ca_cb_length_delta', 0.25))
    max_angle_delta = float(params.get('max_ca_cb_angle_delta_degrees', 20.0))
    min_sep = int(params.get('min_sequence_separation', 3))
    allow_noop = bool(params.get('allow_noop', False))
    require_hbond = bool(params.get('require_hbond', False))
    if not (0 < max_rmsd and 0 < clash_scale <= 1 and min_sep >= 0 and max_bond_delta >= 0 and max_angle_delta >= 0):
        raise ValueError('invalid geometry parameters')
    params.update(max_backbone_rmsd=max_rmsd, clash_radius_scale=clash_scale,
                  min_sequence_separation=min_sep, max_ca_cb_length_delta=max_bond_delta,
                  max_ca_cb_angle_delta_degrees=max_angle_delta, allow_noop=allow_noop,
                  require_hbond=require_hbond)
    model, residues, observed = read_structure(structure_path, chain)
    environment = _environment(model, residues)
    sequence = str(sequence).upper()
    if observed != sequence:
        raise ValueError('structure chain sequence does not exactly match current sequence')
    if len(reference_sequence) != len(sequence):
        raise ValueError('reference sequence length mismatch')
    positions = sorted(set(int(p) for p in mutable_positions))
    if any(p < 0 or p >= len(residues) for p in positions):
        raise ValueError('mutable positions out of range')
    positions = [p for p in positions if sequence[p] not in 'GP' and all(a in residues[p] for a in BACKBONE)]
    libpath = Path(library_path)
    lib = json.loads(libpath.read_text())
    if lib.get('schema_version') != 1 or not isinstance(lib.get('pairs'), list):
        raise ValueError('invalid pair library schema')
    indexed_pairs = [(i, p) for i, p in enumerate(lib['pairs'])
                     if not require_hbond or direct_pair_hbond(p)]
    pairs = [p for _, p in indexed_pairs]
    if not pairs:
        return {'status':'no_library_pairs', 'designed_sequence':sequence, 'metadata':{'considered':0}}
    mobile = np.array([[p[end]['backbone'][a] for end in ('his','partner') for a in BACKBONE] for p in pairs], float)
    library_distances = np.linalg.norm(mobile[:,1] - mobile[:,4], axis=1)
    # If 6-anchor RMSD <= t, CA-CA distance difference is <= sqrt(12)*t.
    # This conservative geometric prefilter cannot drop a passing Kabsch fit.
    candidates = []
    stats = {'library_pairs':len(pairs), 'position_pairs':0, 'distance_candidates':0, 'backbone_fits':0,
             'rejected_attachment':0, 'rejected_clash':0, 'rejected_budget':0, 'rejected_noop':0}
    for pos_h in positions:
        for pos_p in positions:
            if pos_h == pos_p or abs(pos_h-pos_p) < min_sep:
                continue
            stats['position_pairs'] += 1
            target = np.array([residues[pos][a].coord for pos in (pos_h,pos_p) for a in BACKBONE], float)
            distance = np.linalg.norm(target[1]-target[4])
            inds = np.flatnonzero(np.abs(library_distances-distance) <= np.sqrt(12)*max_rmsd + 1e-8)
            stats['distance_candidates'] += len(inds)
            # Batched Kabsch for all library fragments matching the CA-distance bound.
            if not len(inds):
                continue
            x = mobile[inds]
            xc = x - x.mean(axis=1, keepdims=True)
            tc = target - target.mean(axis=0)
            cov = np.einsum('nki,kj->nij', xc, tc)
            u, _, vt = np.linalg.svd(cov)
            d = np.ones((len(inds),3)); d[:,2] = np.linalg.det(u @ vt)
            rot = (u * d[:,None,:]) @ vt
            delta = xc @ rot - tc
            rmsds = np.sqrt(np.mean(np.sum(delta**2, axis=2), axis=1))
            for k in np.flatnonzero(rmsds <= max_rmsd):
                idx = int(inds[k]); pair = pairs[idx]
                aa = str(pair['partner']['aa'])
                if pair['his']['aa'] != 'H' or aa not in 'KRH':
                    raise ValueError('library contains invalid pair type')
                proposed = list(sequence); proposed[pos_h]='H'; proposed[pos_p]=aa
                proposed = ''.join(proposed)
                edits = sum(a!=b for a,b in zip(proposed,sequence))
                total = sum(a!=b for a,b in zip(proposed,reference_sequence))
                if edits > int(max_edits) or total > int(max_total_mutations):
                    stats['rejected_budget'] += 1; continue
                if not edits and not allow_noop:
                    stats['rejected_noop'] += 1; continue
                stats['backbone_fits'] += 1
                candidates.append((float(rmsds[k]), pos_h, pos_p, idx, rot[k], proposed))
    # Rank only by geometrical fit, not by assumed biological efficacy.
    candidates.sort(key=lambda x:(x[0],x[1],x[2],x[3]))
    for rmsd, pos_h, pos_p, idx, rotation, proposed in candidates:
        pair = pairs[idx]
        translation = np.mean([residues[pos][a].coord for pos in (pos_h,pos_p) for a in BACKBONE],axis=0) - mobile[idx].mean(axis=0) @ rotation
        attachment = _attachment(residues,(pos_h,pos_p),pair,rotation,translation)
        if any(a['ca_cb_length_delta_A'] > max_bond_delta or max(a['angle_delta_degrees'].values()) > max_angle_delta for a in attachment):
            stats['rejected_attachment'] += 1; continue
        clash = _clash(environment,residues,(pos_h,pos_p),pair,rotation,translation,clash_scale)
        if clash:
            stats['rejected_clash'] += 1; continue
        placed = {end:{name:(np.asarray(xyz) @ rotation+translation).tolist() for name,xyz in pair[end]['sidechain'].items()} for end in ('his','partner')}
        return {'status':'proposed' if proposed != sequence else 'self_match', 'designed_sequence':proposed,
                'metadata':{'positions_zero_based':[pos_h,pos_p], 'residue_ids':[str(residues[pos_h].id),str(residues[pos_p].id)],
                            'pair_type':'H-'+pair['partner']['aa'], 'backbone_rmsd_A':rmsd,
                            'edits':[{'position_zero_based':i,'from':a,'to':b} for i,(a,b) in enumerate(zip(sequence,proposed)) if a!=b],
                            'source':pair['source'], 'library_index':indexed_pairs[idx][0], 'library_sha256':hashlib.sha256(libpath.read_bytes()).hexdigest(), 'attachment':attachment,
                            'placed_sidechains':placed, 'statistics':stats, 'parameters':params}}
    return {'status':'no_match','designed_sequence':sequence,'metadata':{'statistics':stats,'parameters':params}}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--structure',required=True); ap.add_argument('--chain',required=True)
    ap.add_argument('--library',required=True); ap.add_argument('--mutable',required=True,help='JSON list of zero-based mutable positions')
    ap.add_argument('--reference-sequence'); ap.add_argument('--max-total-mutations',type=int,default=8)
    ap.add_argument('--max-edits',type=int,default=2); ap.add_argument('--max-backbone-rmsd',type=float,default=1.0)
    ap.add_argument('--allow-noop',action='store_true'); ap.add_argument('--output',required=True)
    args=ap.parse_args()
    _,_,seq=read_structure(args.structure,args.chain)
    result=propose_pair(structure_path=args.structure,chain=args.chain,sequence=seq,mutable_positions=json.loads(Path(args.mutable).read_text()),library_path=args.library,reference_sequence=args.reference_sequence or seq,max_total_mutations=args.max_total_mutations,max_edits=args.max_edits,parameters={'max_backbone_rmsd':args.max_backbone_rmsd,'allow_noop':args.allow_noop})
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'metadata':{k:v for k,v in result['metadata'].items() if k!='placed_sidechains'}},indent=2))

if __name__ == '__main__':
    main()
