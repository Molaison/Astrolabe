"""Sequence-position matched CA geometry relative to a fixed supplied complex."""
from functools import lru_cache
from pathlib import Path
import numpy as np


@lru_cache(maxsize=16)
def _coordinates(path):
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
    parser=MMCIFParser(QUIET=True) if Path(path).suffix.lower() in {'.cif','.mmcif'} else PDBParser(QUIET=True)
    model=next(parser.get_structure('complex',str(path)).get_models())
    result={}
    for chain in model:
        residues=[r for r in chain if is_aa(r,standard=True)]
        if residues:
            if any('CA' not in r for r in residues):
                raise ValueError('Missing CA in protein chain')
            result[chain.id]=np.asarray([r['CA'].coord for r in residues],dtype=float)
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite CA coordinates')
    return result


def _fit(reference,moving):
    if reference.shape!=moving.shape or len(reference)<3:
        raise ValueError('Reference/prediction CA correspondence is incomplete')
    u,_,v=np.linalg.svd((moving-moving.mean(0)).T@(reference-reference.mean(0)))
    rotation=u@np.diag([1.,1.,np.linalg.det(u@v)])@v
    return rotation,moving.mean(0),reference.mean(0)


def pose_geometry(reference,predicted,root_reference=None):
    ref=_coordinates(str(Path(reference).resolve()))
    mob=_coordinates(str(Path(predicted).resolve()))
    if any(c not in ref or c not in mob or ref[c].shape!=mob[c].shape for c in ('A','B')):
        raise ValueError('Missing chain or unequal protein residue coverage')
    rotation,center,origin=_fit(ref['B'],mob['B'])
    rms=lambda a,b:float(np.sqrt(np.sum((a-b)**2,axis=1).mean()))
    own,own_center,own_origin=_fit(ref['A'],mob['A'])
    result = dict(target_ca_rmsd=rms(ref['B'],(mob['B']-center)@rotation+origin),
                binder_pose_ca_rmsd=rms(ref['A'],(mob['A']-center)@rotation+origin),
                binder_self_ca_rmsd=rms(ref['A'],(mob['A']-own_center)@own+own_origin))
    if root_reference is not None:
        root = _coordinates(str(Path(root_reference).resolve()))
        rr, rc, ro = _fit(root['B'], mob['B'])
        result['root_binder_pose_ca_rmsd'] = rms(root['A'], (mob['A']-rc)@rr+ro)
        interface = _reference_interface(str(Path(reference).resolve()))
        ref_bb, mob_bb, root_bb = [_backbone_coordinates(str(Path(p).resolve())) for p in (reference,predicted,root_reference)]
        join = lambda value,indices:np.concatenate([value[c][i] for c,i in indices],axis=0)
        def fitted_rms(a,b):
            fit_r,fit_c,fit_o=_fit(a,b)
            return rms(a,(b-fit_c)@fit_r+fit_o)
        result['interface_bb_rmsd'] = fitted_rms(join(ref_bb,interface),join(mob_bb,interface))
        result['root_interface_bb_rmsd'] = fitted_rms(join(root_bb,interface),join(mob_bb,interface))
        all_positions=[(c,i) for c in ('A','B') for i in range(len(ref_bb[c]))]
        result['root_complex_bb_rmsd'] = fitted_rms(join(root_bb,all_positions),join(mob_bb,all_positions))
        tr,tc,to=_fit(root_bb['B'].reshape(-1,3),mob_bb['B'].reshape(-1,3))
        binder_interface=[p for p in interface if p[0]=='A']
        result['root_interface_binder_bb_rmsd']=rms(join(root_bb,binder_interface),(join(mob_bb,binder_interface)-tc)@tr+to)
    return result


@lru_cache(maxsize=64)
def _protein_model(path):
    from Bio.PDB import PDBParser,MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
    parser=MMCIFParser(QUIET=True) if Path(path).suffix.lower() in {'.cif','.mmcif'} else PDBParser(QUIET=True)
    model=next(parser.get_structure('geometry',path).get_models())
    return {c:[r for r in model[c] if is_aa(r,standard=True)] for c in ('A','B')}


@lru_cache(maxsize=64)
def _backbone_coordinates(path):
    value={c:np.asarray([[r[a].coord for a in ('N','CA','C','O')] for r in residues],dtype=float) for c,residues in _protein_model(path).items()}
    if any(not np.isfinite(v).all() for v in value.values()):
        raise ValueError('Nonfinite backbone coordinates')
    return value


@lru_cache(maxsize=16)
def _reference_interface(path):
    from scipy.spatial import cKDTree
    model=_protein_model(path)
    def heavy(chain):
        entries=[(i,a.coord) for i,r in enumerate(model[chain]) for a in r if a.element not in {'H','D'}]
        return np.asarray([e[1] for e in entries]),[e[0] for e in entries]
    left,lp=heavy('A');right,rp=heavy('B')
    neighbors=cKDTree(left).query_ball_tree(cKDTree(right),10.)
    selected={('A',lp[i]) for i,matches in enumerate(neighbors) if matches}
    selected.update(('B',rp[j]) for matches in neighbors for j in matches)
    if not selected:raise ValueError('No original reference interface within10A')
    return tuple(sorted(selected))


def verify_prediction_sequences(predicted,binder_sequence,target_sequence):
    from Bio.PDB import PDBParser,MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.SeqUtils import seq1
    path=Path(predicted)
    parser=MMCIFParser(QUIET=True) if path.suffix.lower() in {'.cif','.mmcif'} else PDBParser(QUIET=True)
    model=next(parser.get_structure('sequence_check',str(path)).get_models())
    for chain,expected in [('A',binder_sequence),('B',target_sequence)]:
        if chain not in model:
            raise ValueError('Predicted chain is missing')
        actual=''.join(seq1(r.resname) for r in model[chain] if is_aa(r,standard=True))
        if actual!=expected:
            raise ValueError(f'Predicted chain {chain} sequence differs from the requested sequence')
