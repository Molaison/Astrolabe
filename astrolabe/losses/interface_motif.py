"""Frozen binder interface geometry, usable with binder-only predictions."""
from pathlib import Path

import numpy as np

from astrolabe.losses.base import loss_registry


def _model(path):
    from astrolabe.losses.design_screening import _load_biopython_structure
    return next(_load_biopython_structure(Path(path)).get_models())


def _residues(model, chain):
    from Bio.PDB.Polypeptide import is_aa
    if chain not in model:
        raise ValueError(f"Chain {chain!r} is missing from structure.")
    result = [r for r in model[chain] if is_aa(r, standard=True)]
    if not result:
        raise ValueError(f"Chain {chain!r} contains no standard protein residues.")
    return result


def _coordinates(atoms):
    result = np.asarray([a.coord for a in atoms], dtype=float)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError("Selected atoms must contain finite coordinates.")
    return result


def _interface(model, binder_chain, target_chains, cutoff):
    from scipy.spatial import cKDTree
    cutoff = float(cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("contact_cutoff must be positive and finite (angstrom).")
    binder = _residues(model, binder_chain)
    if target_chains is None:
        target_chains = [chain.id for chain in model if chain.id != binder_chain]
    elif isinstance(target_chains, str):
        target_chains = [target_chains]
    else:
        target_chains = list(target_chains)
    if not target_chains or binder_chain in target_chains:
        raise ValueError("Reference target chains must be nonempty and disjoint from the binder.")
    target_chains = list(dict.fromkeys(target_chains))
    targets = [res for chain in target_chains for res in _residues(model, chain)]
    heavy = lambda res: [a for a in res if (a.element or '').strip().upper() not in {'H', 'D'}]
    target_atoms = [a for res in targets for a in heavy(res)]
    tree = cKDTree(_coordinates(target_atoms))
    positions = []
    for i, res in enumerate(binder):
        if any(tree.query_ball_point(_coordinates(heavy(res)), cutoff)):
            positions.append(i)
    if not positions:
        raise ValueError("No reference interface residues within contact_cutoff.")
    return binder, tuple(positions)


def reference_interface_positions(reference_complex, reference_binder_chain='A',
                                  reference_target_chains=None, contact_cutoff=4.5):
    """Return frozen zero-based binder sequence positions from reference contacts."""
    _, positions = _interface(_model(reference_complex), reference_binder_chain,
                              reference_target_chains, contact_cutoff)
    return positions


@loss_registry.register(
    name='interface_motif_rmsd',
    required_params=['reference_complex', 'binder_chain'],
    optional_params=['reference_binder_chain', 'reference_target_chains', 'contact_cutoff'],
    description='Frozen binder-interface N/CA/C/O RMSD after one joint motif fit (angstrom)',
    supports_oligomer_specific=True,
)
def interface_motif_rmsd(eval_result, reference_complex, binder_chain,
                         reference_binder_chain='A', reference_target_chains=None,
                         contact_cutoff=4.5):
    """Compare by chain sequence order; only the reference needs target chains."""
    from Bio.PDB import Superimposer
    from astrolabe.core.chain_resolution import resolve_selector_chain_ids

    reference, positions = _interface(_model(reference_complex), reference_binder_chain,
                                      reference_target_chains, contact_cutoff)
    scores = {}
    for name in eval_result.names:
        chains = resolve_selector_chain_ids(name, binder_chain)
        if len(chains) != 1:
            raise ValueError('binder_chain must resolve to exactly one prediction chain.')
        prediction = _residues(_model(eval_result.complex_paths[name]), chains[0])
        if len(prediction) != len(reference):
            raise ValueError('Reference and predicted binder must have equal residue coverage.')
        fixed, moving = [], []
        for i in positions:
            ref, pred = reference[i], prediction[i]
            if ref.resname != pred.resname:
                raise ValueError(f'Frozen interface amino acid changed at binder position {i+1}.')
            for atom_name in ('N', 'CA', 'C', 'O'):
                if atom_name not in ref or atom_name not in pred:
                    raise ValueError(f'Missing backbone atom {atom_name!r} at interface position {i+1}.')
                fixed.append(ref[atom_name]); moving.append(pred[atom_name])
        _coordinates(fixed); _coordinates(moving)
        fit = Superimposer()
        fit.set_atoms(fixed, moving)
        scores[name] = float(fit.rms)
    return scores


@loss_registry.register(
    name='interface_motif_rmsd_norm',
    required_params=['reference_complex', 'binder_chain'],
    optional_params=['reference_binder_chain', 'reference_target_chains', 'contact_cutoff', 'best', 'worst'],
    description='Normalized frozen binder-interface geometry score; higher is better',
    supports_oligomer_specific=True,
)
def interface_motif_rmsd_norm(eval_result, reference_complex, binder_chain,
                              reference_binder_chain='A', reference_target_chains=None,
                              contact_cutoff=4.5, best=0.0, worst=2.0):
    best, worst = float(best), float(worst)
    if not np.isfinite([best, worst]).all() or best < 0 or worst <= best:
        raise ValueError('Normalization requires finite 0 <= best < worst (angstrom).')
    raw = interface_motif_rmsd(eval_result, reference_complex, binder_chain,
                               reference_binder_chain, reference_target_chains, contact_cutoff)
    return {name: float(np.clip((worst-value)/(worst-best), 0., 1.))
            for name, value in raw.items()}
