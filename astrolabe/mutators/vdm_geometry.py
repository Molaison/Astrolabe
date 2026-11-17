#!/usr/bin/env python3
"""Two-site, chain-aware natural vdM geometry search; no runtime integration.

Positions are *polymer order*, not PDB residue numbers: sites use 1-based
positions, mutable_positions uses binder 0-based positions. A null position
enumerates the specified chain, including fixed-sequence target chains.
Only (mutable, template), (fixed, fixed), and (fixed, repack) policies are legal.
An explicit fixed identity mismatch is an input error; unavailable/protected
mutable positions simply cannot produce a match. The supplied mask is an upper
bound: protection parameters can narrow it, never expand it.

Repack samples complete same-identity natural sidechains from this library,
individually fitted to the actual site's N/CA/C. It is NOT Dunbrack repacking.
Fixed sidechains use an all-heavy-atom name correspondence, with only the ARG
NH1/NH2 graph symmetry allowed. Histidine ring flips are conservatively rejected.
Reciprocal source annotations identify donor/acceptor edges, but acceptance
ALWAYS measures those edges in the final coordinates. Heavy-atom proximity is
not a protonation assignment or a hydrogen-bond angle/energy calculation.

The defaults retain v1 geometry (1 A, .25 A, 20 degrees, .75 vdW, gap 3).
Cys/Gly/Pro protections are opt-in and only narrow the caller's effective mask.
No candidate/rotamer truncation, file writes, downloads, or AF3 calls occur in
propose_pair. write_preview is an explicit optional coordinate export helper.
"""
import copy
import hashlib
import itertools
import json
import math
import numbers
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFIO, PDBIO
from Bio.PDB.Atom import Atom
from Bio.SeqUtils import seq1, seq3
from scipy.spatial import cKDTree

# Retain the validated v1 geometry primitives within this native package.
from astrolabe.mutators.vdm_geometry_legacy import fit, read_structure, BACKBONE, MAINCHAIN, RADII
CANONICAL = frozenset('ACDEFGHIKLMNPQRSTVWY')
SIDECHAINS = {
    'H': ('CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'),
    'K': ('CB', 'CG', 'CD', 'CE', 'NZ'),
    'R': ('CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'),
    'D': ('CB', 'CG', 'OD1', 'OD2'),
    'E': ('CB', 'CG', 'CD', 'OE1', 'OE2'),
}
HYDROGEN_PARENT = {
    'H': {'HD1': 'ND1', 'HE2': 'NE2'},
    'K': {'HZ1': 'NZ', 'HZ2': 'NZ', 'HZ3': 'NZ'},
    'R': {'HE': 'NE', 'HH11': 'NH1', 'HH12': 'NH1',
          'HH21': 'NH2', 'HH22': 'NH2'},
    'D': {}, 'E': {},
}
ACCEPTORS = {'H': ('ND1', 'NE2'), 'D': ('OD1', 'OD2'), 'E': ('OE1', 'OE2')}
DEFAULTS = dict(max_backbone_rmsd=1.0, clash_radius_scale=0.75,
                max_ca_cb_length_delta=0.25, max_ca_cb_angle_delta_degrees=20.0,
                min_sequence_separation=3, max_fixed_sidechain_rmsd=0.5,
                max_hbond_distance=3.5, max_repack_backbone_rmsd=1.0,
                protect_gly_pro=False, preserve_disulfides=False,
                require_hbond=True, allow_noop=False, partner_types=('H', 'K', 'R'))


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return int(value)


def _sequence(value, label):
    if not isinstance(value, str) or not value or any(a not in CANONICAL for a in value):
        raise ValueError(f'{label} must be a nonempty uppercase canonical protein sequence')
    return value


def _parameters(parameters):
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError('parameters must be a dict')
    params = dict(DEFAULTS)
    unknown = set(parameters or {}) - set(params)
    if unknown:
        raise ValueError(f'unknown parameters: {sorted(unknown)}')
    params.update(parameters or {})
    types = params['partner_types']
    if isinstance(types, str):
        types = [types]
    if not isinstance(types, (list, tuple)) or not types or any(x not in SIDECHAINS for x in types):
        raise ValueError('partner_types must be a nonempty list drawn from H, K, R, D, E')
    params['partner_types'] = sorted(set(types))
    for key in ('protect_gly_pro', 'preserve_disulfides', 'require_hbond', 'allow_noop'):
        if not isinstance(params[key], bool):
            raise ValueError(f'{key} must be boolean')
    if params['allow_noop'] or not params['require_hbond']:
        raise ValueError('nonempty mutations and actual direct-pair distance checks are mandatory')
    params['min_sequence_separation'] = _integer(params['min_sequence_separation'], 'min_sequence_separation')
    for key in set(params) - {'min_sequence_separation', 'protect_gly_pro',
                             'preserve_disulfides', 'require_hbond', 'allow_noop', 'partner_types'}:
        v = params[key]
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v) or v < 0:
            raise ValueError(f'invalid geometry parameter {key}')
        params[key] = float(v)
    if (params['max_backbone_rmsd'] <= 0 or params['max_hbond_distance'] <= 0 or
            not 0 < params['clash_radius_scale'] <= 1):
        raise ValueError('RMSD/distance must be positive; clash scale must be in (0, 1]')
    return params


def _xyz(value):
    xyz = np.asarray(value, float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError('coordinates must be finite 3-vectors')
    return xyz


def _heavy(residue):
    return {a.name: np.asarray(a.coord, float).copy() for a in residue
            if a.name not in MAINCHAIN and a.element.upper() not in ('H', 'D')}


def _usable(residue):
    # Do not choose an ambiguous participating conformer implicitly.
    return (not residue.is_disordered() and all(a in residue for a in BACKBONE)
            and all(np.isfinite(residue[a].coord).all() for a in BACKBONE)
            and np.linalg.matrix_rank(np.array([residue[a].coord for a in BACKBONE])
                                      - residue['CA'].coord) >= 2)


def _edges(pair):
    """Return reciprocal annotated donor-heavy -> acceptor-heavy edges.

    X is partner; Y is His in the v1 library. Only chemically supported sidechain
    hydrogen/nitrogen contacts qualify; boolean `source.hbond` is never used.
    """
    contacts = {}
    for row in pair.get('source', {}).get('hbond_contacts', []):
        key = (row.get('chain'), row.get('name'))
        contacts.setdefault(key, set()).update(
            x.strip() for x in (row.get('partners_hb') or '').split(',') if x.strip())
    edges = set()
    roles = {'X': 'partner', 'Y': 'his'}
    for donor_chain, acceptor_chain in (('X', 'Y'), ('Y', 'X')):
        donor_role, acceptor_role = roles[donor_chain], roles[acceptor_chain]
        aa = pair[donor_role]['aa']
        acceptors = ACCEPTORS.get(pair[acceptor_role]['aa'], ())
        if not acceptors:
            continue
        for hydrogen, donor in HYDROGEN_PARENT[aa].items():
            for acceptor in acceptors:
                if (acceptor in contacts.get((donor_chain, hydrogen), set()) and
                        hydrogen in contacts.get((acceptor_chain, acceptor), set())):
                    edges.add((donor_role, donor, acceptor_role, acceptor))
    return sorted(edges)


def _library(path):
    raw = Path(path).read_bytes()
    lib = json.loads(raw)
    if not isinstance(lib, dict) or lib.get('schema_version') != 1 or not isinstance(lib.get('pairs'), list):
        raise ValueError('invalid pair library schema')
    for index, pair in enumerate(lib['pairs']):
        try:
            if pair['his']['aa'] != 'H' or pair['partner']['aa'] not in SIDECHAINS:
                raise ValueError('invalid H--K/R/H/D/E pair identity')
            for role in ('his', 'partner'):
                unit = pair[role]
                if set(unit['sidechain']) != set(SIDECHAINS[unit['aa']]):
                    raise ValueError('library requires complete canonical heavy sidechains')
                for atom in BACKBONE:
                    _xyz(unit['backbone'][atom])
                for coord in unit['sidechain'].values():
                    _xyz(coord)
                bb = np.array([unit['backbone'][a] for a in BACKBONE])
                if np.linalg.matrix_rank(bb-bb.mean(0)) < 2:
                    raise ValueError('degenerate library backbone')
            _edges(pair)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'invalid library pair {index}: {exc}') from exc
    return lib['pairs'], hashlib.sha256(raw).hexdigest()


def _sites(sites, binder_chain, chains, effective):
    if not isinstance(sites, list) or len(sites) != 2:
        raise ValueError('sites must be a length-2 list')
    normalized, choices = [], []
    required = {'chain', 'position', 'residue_type', 'sequence', 'sidechain'}
    for index, site in enumerate(sites):
        if not isinstance(site, dict) or set(site) != required:
            raise ValueError(f'site {index} must explicitly provide {sorted(required)}')
        site = dict(site)
        chain = site['chain']
        if not isinstance(chain, str) or chain not in chains:
            raise ValueError(f'site {index}: unknown real structure chain {chain!r}')
        mode, policy = site['sequence'], site['sidechain']
        if (mode, policy) not in {('mutable', 'template'), ('fixed', 'fixed'), ('fixed', 'repack')}:
            raise ValueError('legal sequence/sidechain policies: mutable/template, fixed/fixed, fixed/repack')
        if mode == 'mutable' and chain != binder_chain:
            raise ValueError('only binder_chain may have mutable sequence')
        types_ = site['residue_type']
        if isinstance(types_, str):
            types_ = [types_]
        if types_ is not None and (not isinstance(types_, list) or not types_ or
                                  any(not isinstance(a, str) or len(a) != 1 or a not in CANONICAL for a in types_)):
            raise ValueError('residue_type must be null, one uppercase letter, or a nonempty list of letters')
        site['residue_type'] = None if types_ is None else sorted(set(types_))
        residues, sequence = chains[chain]
        pos = site['position']
        if pos is None:
            positions = range(len(residues))
        else:
            pos = _integer(pos, f'site {index} position', minimum=1)
            if pos > len(residues):
                raise ValueError(f'site {index} position outside chain')
            site['position'] = pos
            positions = [pos-1]
            if mode == 'fixed' and types_ is not None and sequence[pos-1] not in types_:
                raise ValueError(f'fixed site {index} residue_type disagrees with structure identity {sequence[pos-1]}')
        selected = []
        for p in positions:
            if mode == 'mutable' and p not in effective:
                continue
            if mode == 'fixed' and types_ is not None and sequence[p] not in types_:
                continue
            if not _usable(residues[p]):
                continue
            selected.append(p)
        normalized.append(site)
        choices.append(selected)
    if (normalized[0]['chain'] == normalized[1]['chain'] and
            normalized[0]['position'] is not None and
            normalized[0]['position'] == normalized[1]['position']):
        raise ValueError('the two sites cannot be the same residue')
    return normalized, choices


def _attachment(residue, unit, coords):
    def angle(a, b):
        denom = np.linalg.norm(a)*np.linalg.norm(b)
        if denom <= 1e-12:
            return float('inf')
        return float(np.degrees(np.arccos(np.clip(np.dot(a, b)/denom, -1, 1))))
    ca, cb = _xyz(unit['backbone']['CA']), _xyz(unit['sidechain']['CB'])
    source = cb-ca
    target = coords['CB']-residue['CA'].coord
    length = abs(float(np.linalg.norm(target)-np.linalg.norm(source)))
    angles = {a: abs(angle(_xyz(unit['backbone'][a])-ca, source) -
                     angle(residue[a].coord-residue['CA'].coord, target)) for a in ('N', 'C')}
    return dict(ca_cb_length_delta_A=length, angle_delta_degrees=angles)


def _attachment_ok(record, params):
    values = [record['ca_cb_length_delta_A'], *record['angle_delta_degrees'].values()]
    return (all(math.isfinite(x) for x in values) and
            values[0] <= params['max_ca_cb_length_delta'] and
            max(values[1:]) <= params['max_ca_cb_angle_delta_degrees'])


def _fixed_fit(unit, template, actual):
    if set(actual) != set(unit['sidechain']):
        return None  # No partial-atom RMSD or ad-hoc histidine atom swaps.
    identity = {name: name for name in template}
    mappings = [identity]
    if unit['aa'] == 'R':
        swapped = dict(identity, NH1='NH2', NH2='NH1')
        mappings.append(swapped)
    trials = [(float(np.sqrt(np.mean([np.sum((template[a]-actual[m[a]])**2)
                                        for a in template]))), m) for m in mappings]
    return min(trials, key=lambda x: x[0])


def _environment(model, chains):
    keys = {id(r): (chain, p) for chain, (residues, _) in chains.items()
            for p, r in enumerate(residues)}
    env = []
    for chain in model:
        for residue in chain.get_unpacked_list():
            for atom in residue.get_unpacked_list():
                element = (atom.element or atom.name[0]).upper()
                if element not in RADII:
                    continue
                xyz = _xyz(atom.coord)
                env.append((xyz, element, keys.get(id(residue)), atom.name))
    tree = cKDTree(np.array([row[0] for row in env]).reshape(-1, 3))
    return env, tree


def _clash(environment, keys, options, scale):
    env, tree = environment
    # Both pair sidechains are represented below, including untouched fixed ones.
    # Keep their real backbones; residue identity includes the chain.
    placed = [(xyz, name[0], key, name) for key, option in zip(keys, options)
              for name, xyz in option['coords'].items()]
    for xyz, element, key, name in placed:
        for i in tree.query_ball_point(xyz, scale*(RADII[element]+max(RADII.values()))):
            other, e2, other_key, other_name = env[i]
            if other_key == key or (other_key in keys and other_name not in MAINCHAIN):
                continue
            d = float(np.linalg.norm(xyz-other))
            if d < scale*(RADII[element]+RADII[e2]):
                return True
    for i, (xyz, element, key, _) in enumerate(placed):
        for other, e2, key2, _ in placed[i+1:]:
            if key != key2 and np.linalg.norm(xyz-other) < scale*(RADII[element]+RADII[e2]):
                return True
    return False


def _distances(edges, roles, options, keys):
    output = []
    for donor_role, donor_atom, acceptor_role, acceptor_atom in edges:
        i, j = roles.index(donor_role), roles.index(acceptor_role)
        donor_atom = options[i]['atom_mapping'][donor_atom]
        acceptor_atom = options[j]['atom_mapping'][acceptor_atom]
        output.append(dict(donor_site_index=i, donor_chain=keys[i][0], donor_atom=donor_atom,
                           acceptor_site_index=j, acceptor_chain=keys[j][0], acceptor_atom=acceptor_atom,
                           distance_A=float(np.linalg.norm(options[i]['coords'][donor_atom] -
                                                           options[j]['coords'][acceptor_atom]))))
    return output


def propose_pair(*, structure_path, binder_chain, binder_sequence, mutable_positions,
                 library_path, reference_sequence, max_total_mutations, max_edits,
                 sites, parameters=None, excluded_aas=()):
    """Return {status: proposed|no_match, designed_sequence, metadata}.

    Invalid contracts raise ValueError. Exhausted valid searches return no_match,
    never an unmutated self-match. Ranking is global six-backbone-atom RMSD first;
    repack alternatives are subsequently tried by their local backbone RMSD.
    """
    return _search_pair_geometry(
        structure_path=structure_path, binder_chain=binder_chain,
        binder_sequence=binder_sequence, mutable_positions=mutable_positions,
        library_path=library_path, reference_sequence=reference_sequence,
        max_total_mutations=max_total_mutations, max_edits=max_edits,
        sites=sites, parameters=parameters, excluded_aas=excluded_aas)


def propose_pair_including_existing(**kwargs):
    """Native-stage search also accepts a geometrically valid unchanged pair."""
    return _search_pair_geometry(**kwargs, _accept_unchanged=True)


def _search_pair_geometry(*, structure_path, binder_chain, binder_sequence,
                          mutable_positions, library_path, reference_sequence,
                          max_total_mutations, max_edits, sites, parameters=None,
                          excluded_aas=(), _pair_set=None, _accept_unchanged=False):
    """Shared original-input search; optionally retain bounded valid options."""
    params = _parameters(parameters)
    sequence = _sequence(binder_sequence, 'binder_sequence')
    reference = _sequence(reference_sequence, 'reference_sequence')
    if len(sequence) != len(reference):
        raise ValueError('reference sequence length mismatch')
    max_total_mutations = _integer(max_total_mutations, 'max_total_mutations')
    max_edits = _integer(max_edits, 'max_edits')
    if _pair_set is None and max_edits > 2:
        raise ValueError('max_edits must be <= 2 for a pair proposal')
    excluded_aas = set(excluded_aas)
    if excluded_aas - CANONICAL:
        raise ValueError('excluded_aas must contain canonical one-letter identities')
    model, _, observed = read_structure(structure_path, binder_chain)
    if observed != sequence:
        raise ValueError('structure binder sequence does not exactly match binder_sequence')
    chains = {}
    for chain in model:
        residues = [r for r in chain if r.id[0] == ' ']
        chains[chain.id] = (residues, ''.join(seq1(r.resname, custom_map={'UNK': 'X'}) for r in residues))
    try:
        mask = {_integer(p, 'mutable_positions entry') for p in mutable_positions}
    except TypeError as exc:
        raise ValueError('mutable_positions must be an iterable of 0-based integer indices') from exc
    if any(p >= len(sequence) for p in mask):
        raise ValueError('mutable_positions outside binder')
    protected = {p for p in mask if (params['protect_gly_pro'] and sequence[p] in 'GP') or
                 (params['preserve_disulfides'] and sequence[p] == 'C')}
    effective = mask-protected
    specs, choices = _sites(sites, binder_chain, chains, effective)
    pairs, library_sha = _library(library_path)
    stats = {name: 0 for name in (
        'position_pairs', 'rejected_duplicate_site', 'rejected_separation', 'rejected_identity',
        'rejected_excluded_aa',
        'distance_candidates', 'backbone_fits', 'rejected_backbone', 'rejected_budget',
        'rejected_noop', 'rejected_source_hbond', 'rejected_fixed_sidechain',
        'rejected_attachment', 'rejected_clash', 'rejected_hbond',
        'repack_templates_considered', 'rejected_repack_backbone')}
    stats['library_pairs'] = len(pairs)
    stats['matching_library_pairs'] = sum(pair['partner']['aa'] in params['partner_types'] for pair in pairs)
    if pairs and not stats['matching_library_pairs']:
        raise ValueError('vdm_pair library contains none of the requested partner_types')
    target_sequences = {chain: seq for chain, (_, seq) in chains.items() if chain != binder_chain}
    metadata = dict(parameters=params, statistics=stats, library_sha256=library_sha,
                    structure_sha256=hashlib.sha256(Path(structure_path).read_bytes()).hexdigest(),
                    binder_chain=binder_chain, original_binder_sequence=sequence,
                    input_sites=specs, available_positions_per_site=list(map(len, choices)),
                    effective_mutable_positions=sorted(effective),
                    protected_mutable_positions=sorted(protected),
                    target_sequences_before=target_sequences,
                    target_sequences_after=dict(target_sequences), target_sequence_unchanged=True,
                    coordinate_delivery='preview coordinates only; no Astrolabe/AF3 target-coordinate integration',
                    fixed_atom_equivalence='identity names plus ARG NH1/NH2 graph symmetry only',
                    hbond_interpretation='reciprocal annotated edges measured on final heavy atoms; no protonation/angle claim',
                    repack_sampling='same-identity natural library templates on actual N/CA/C; not full Dunbrack repack',
                    placed_sidechains=[], sites=[], direct_pair_hbonds=[], source=None)
    # Expand both library role assignments. Identity filtering precedes fitting.
    records = []
    for index, pair in enumerate(pairs):
        if pair['partner']['aa'] not in params['partner_types']:
            continue
        edges = _edges(pair)
        if not edges:
            stats['rejected_source_hbond'] += 1
            continue
        for roles in (('his', 'partner'), ('partner', 'his')):
            mobile = np.array([pair[role]['backbone'][a] for role in roles for a in BACKBONE], float)
            records.append((index, roles, mobile, edges))
    candidates = []
    for positions in itertools.product(*choices):
        keys = tuple((spec['chain'], p) for spec, p in zip(specs, positions))
        if keys[0] == keys[1]:
            stats['rejected_duplicate_site'] += 1
            continue
        if keys[0][0] == keys[1][0] and abs(positions[0]-positions[1]) < params['min_sequence_separation']:
            stats['rejected_separation'] += 1
            continue
        stats['position_pairs'] += 1
        residues = [chains[c][0][p] for c, p in keys]
        actual_types = [chains[c][1][p] for c, p in keys]
        target = np.array([r[a].coord for r in residues for a in BACKBONE], float)
        ca_distance = np.linalg.norm(target[1]-target[4])
        selected = []
        for record in records:
            index, roles, mobile, edges = record
            identities = [pairs[index][role]['aa'] for role in roles]
            if any(spec['sequence'] == 'mutable' and aa != actual and aa in excluded_aas
                   for spec, aa, actual in zip(specs, identities, actual_types)):
                stats['rejected_excluded_aa'] += 1
                continue
            if any((spec['residue_type'] is not None and aa not in spec['residue_type']) or
                   (spec['sequence'] == 'fixed' and aa != actual)
                   for spec, aa, actual in zip(specs, identities, actual_types)):
                stats['rejected_identity'] += 1
                continue
            if abs(np.linalg.norm(mobile[1]-mobile[4])-ca_distance) <= math.sqrt(12)*params['max_backbone_rmsd']+1e-8:
                selected.append(record)
        stats['distance_candidates'] += len(selected)
        if not selected:
            continue
        # v1 batched proper Kabsch: six N/CA/C anchors, one rigid pair transform.
        mobiles = np.array([r[2] for r in selected])
        xc, tc = mobiles-mobiles.mean(1, keepdims=True), target-target.mean(0)
        u, _, vt = np.linalg.svd(np.einsum('nki,kj->nij', xc, tc))
        d = np.ones((len(selected), 3)); d[:, 2] = np.linalg.det(u @ vt)
        rotations = (u*d[:, None, :]) @ vt
        rmsds = np.sqrt(np.mean(np.sum((xc @ rotations-tc)**2, axis=2), axis=1))
        stats['backbone_fits'] += len(selected)
        stats['rejected_backbone'] += int(np.count_nonzero(rmsds > params['max_backbone_rmsd']))
        for k in np.flatnonzero(rmsds <= params['max_backbone_rmsd']):
            index, roles, mobile, edges = selected[k]
            proposed = list(sequence)
            for spec, p, role in zip(specs, positions, roles):
                if spec['sequence'] == 'mutable':
                    proposed[p] = pairs[index][role]['aa']
            proposed = ''.join(proposed)
            edits = sum(a != b for a, b in zip(sequence, proposed))
            if not edits and _pair_set is None and not _accept_unchanged:
                stats['rejected_noop'] += 1
                continue
            # Another disjoint pair may restore parent-reference identities.
            # Single-pair mode retains the historical exact budget filter.
            recoverable = (0 if _pair_set is None else
                           min(max_edits-edits, 2*(_pair_set['pair_count']-1)))
            if (edits > max_edits or
                    sum(a != b for a, b in zip(reference, proposed)) > max_total_mutations+recoverable):
                stats['rejected_budget'] += 1
                continue
            rotation = rotations[k]
            translation = target.mean(0)-mobile.mean(0) @ rotation
            candidates.append((float(rmsds[k]), keys, index, roles, rotation, translation, proposed, edges))
    candidates.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
    metadata['ranked_candidates'] = len(candidates)
    if not candidates:
        return dict(status='no_match', designed_sequence=sequence, metadata=metadata)
    environment = _environment(model, chains)
    if _pair_set is not None:
        _pair_set['environment'] = environment
    # Each real site/type is locally fitted once, independent of pair transform.
    repack_cache = {}

    def repack_options(key, aa):
        cache_key = (*key, aa)
        if cache_key not in repack_cache:
            residue = chains[key[0]][0][key[1]]
            target = np.array([residue[a].coord for a in BACKBONE])
            options = []
            for index, pair in enumerate(pairs):
                if pair['partner']['aa'] not in params['partner_types']:
                    continue
                for role in ('his', 'partner'):
                    unit = pair[role]
                    if unit['aa'] != aa:
                        continue
                    stats['repack_templates_considered'] += 1
                    r, t, rmsd = fit([unit['backbone'][a] for a in BACKBONE], target)
                    if rmsd > params['max_repack_backbone_rmsd']:
                        stats['rejected_repack_backbone'] += 1
                        continue
                    coords = {a: _xyz(x) @ r+t for a, x in unit['sidechain'].items()}
                    attachment = _attachment(residue, unit, coords)
                    if not _attachment_ok(attachment, params):
                        stats['rejected_attachment'] += 1
                        continue
                    options.append(dict(coords=coords, atom_mapping={a: a for a in coords},
                                        attachment=attachment, local_backbone_rmsd_A=rmsd,
                                        sampling_source=dict(library_index=index, role=role, source=pair.get('source', {}))))
            options.sort(key=lambda x: (x['local_backbone_rmsd_A'], x['sampling_source']['library_index'], x['sampling_source']['role']))
            repack_cache[cache_key] = options
        return repack_cache[cache_key]

    for rmsd, keys, index, roles, rotation, translation, proposed, edges in candidates:
        pair = pairs[index]
        option_lists = []
        for spec, key, role in zip(specs, keys, roles):
            residue = chains[key[0]][0][key[1]]
            unit = pair[role]
            if spec['sidechain'] == 'repack':
                option_lists.append(repack_options(key, unit['aa']))
                continue
            template = {a: _xyz(x) @ rotation+translation for a, x in unit['sidechain'].items()}
            atom_mapping, fixed_rmsd = {a: a for a in template}, None
            coords = template
            if spec['sidechain'] == 'fixed':
                coords = _heavy(residue)
                matched = _fixed_fit(unit, template, coords)
                if matched is None or matched[0] > params['max_fixed_sidechain_rmsd']:
                    stats['rejected_fixed_sidechain'] += 1
                    option_lists.append([])
                    continue
                fixed_rmsd, atom_mapping = matched
            attachment = _attachment(residue, unit, coords)
            if not _attachment_ok(attachment, params):
                stats['rejected_attachment'] += 1
                option_lists.append([])
                continue
            option_lists.append([dict(coords=coords, atom_mapping=atom_mapping,
                                      attachment=attachment, fixed_sidechain_rmsd_A=fixed_rmsd)])
        endpoint_identity = tuple(sorted((key[0], key[1], pair[role]['aa'])
                                         for key, role in zip(keys, roles)))
        for options in itertools.product(*option_lists):
            if (_pair_set is not None and
                    _pair_set['endpoint_identity_counts'].get(endpoint_identity, 0) >=
                    _pair_set['endpoint_identity_option_cap']):
                # Count skipped option combinations, not validated pairs: these
                # alternatives are deliberately not geometry-tested after the
                # first four accepted options for these endpoints/identities.
                _pair_set['skipped_endpoint_identity_options'] += 1
                _pair_set['skipped_endpoint_identity_groups'].add(endpoint_identity)
                _pair_set['search_exhausted'] = False
                continue
            distances = _distances(edges, roles, options, keys)
            if not distances or any(d['distance_A'] > params['max_hbond_distance'] for d in distances):
                stats['rejected_hbond'] += 1
                continue
            if _clash(environment, keys, options, params['clash_radius_scale']):
                stats['rejected_clash'] += 1
                continue
            site_records, placed = [], []
            for i, (spec, key, role, option) in enumerate(zip(specs, keys, roles, options)):
                residue = chains[key[0]][0][key[1]]
                aa = pair[role]['aa']
                record = dict(site_index=i, chain=key[0], position_one_based=key[1]+1,
                              position_zero_based=key[1], residue_id=list(residue.id),
                              original_residue_type=chains[key[0]][1][key[1]], residue_type=aa,
                              sequence=spec['sequence'], sidechain_policy=spec['sidechain'],
                              library_role=role)
                site_records.append(record)
                entry = dict(record, coordinates={a: xyz.tolist() for a, xyz in option['coords'].items()},
                             atom_mapping=dict(option['atom_mapping']), attachment=option['attachment'])
                entry.update({k: v for k, v in option.items() if k not in ('coords', 'atom_mapping', 'attachment')})
                original = _heavy(residue)
                entry['max_original_atom_displacement_A'] = max(
                    (float(np.linalg.norm(xyz-original[a])) for a, xyz in option['coords'].items() if a in original), default=0.0)
                placed.append(entry)
            metadata.update(sites=site_records, placed_sidechains=placed, source=pair.get('source', {}),
                            library_index=index, library_roles=list(roles),
                            mode=[s['sidechain'] for s in specs], backbone_rmsd_A=rmsd,
                            pair_transform=dict(convention='row_vector: xyz @ rotation + translation',
                                                rotation=rotation.tolist(), translation=translation.tolist()),
                            direct_pair_hbonds=distances,
                            edits=[dict(position_zero_based=p, position_one_based=p+1, **{'from': a, 'to': b})
                                   for p, (a, b) in enumerate(zip(sequence, proposed)) if a != b],
                            total_mutations=sum(a != b for a, b in zip(reference, proposed)),
                            fixed_site_identities_unchanged=all(s['residue_type'] == s['original_residue_type']
                                                             for s in site_records if s['sequence'] == 'fixed'))
            result = dict(status='proposed', designed_sequence=proposed, metadata=metadata)
            if _pair_set is None:
                return result
            # Each snapshot retains the complete v1 metadata, including edits
            # against the original parent, never edits from a sequential pose.
            _pair_set['options'].append(copy.deepcopy(result))
            counts = _pair_set['endpoint_identity_counts']
            counts[endpoint_identity] = counts.get(endpoint_identity, 0)+1
            if len(_pair_set['options']) == _pair_set['candidate_cap']:
                # No lookahead: even if this happened to be the final option,
                # completion of the unvisited search has not been established.
                _pair_set['candidate_cap_reached'] = True
                _pair_set['search_exhausted'] = False
                return result
    return dict(status='no_match', designed_sequence=sequence, metadata=metadata)


def propose_pair_set(*, structure_path, binder_chain, binder_sequence,
                     mutable_positions, library_path, reference_sequence,
                     max_total_mutations, max_edits, sites, parameters=None,
                     excluded_aas=(), pair_count=2):
    """Propose exactly N disjoint pairs from bounded original-input geometry.

    At most 128 validated options (including distinct repack alternatives) are
    retained in the single-pair ranking order, at most four per unordered set of
    (physical chain, zero-based polymer position, proposed residue identity).
    Further option combinations for a saturated group are skipped before their
    geometry checks; their count and the number of affected groups are recorded.
    Lexicographic backtracking visits at most 100000 candidate extensions,
    counting rejected extensions as well. Any cap that truncates either search
    marks it incomplete. No partial sequence is ever returned. Unchanged pairs
    are eligible, and even a zero-edit aggregate can return proposed.

    Each pair is independently validated against the unchanged input structure;
    this conservatively excludes pairs clashing with an original sidechain that
    a different pair could have moved. Combined geometry is checked again with
    all selected sidechains present. No structure or library is written.
    """
    pair_count = _integer(pair_count, 'pair_count', minimum=1)
    max_edits = _integer(max_edits, 'max_edits')
    if max_edits > 2*pair_count:
        raise ValueError('max_edits must be <= 2*pair_count for a pair-set proposal')
    max_total_mutations = _integer(max_total_mutations, 'max_total_mutations')
    search = dict(pair_count=pair_count, candidate_cap=128, options=[],
                  candidate_cap_reached=False, search_exhausted=True,
                  endpoint_identity_option_cap=4, endpoint_identity_counts={},
                  skipped_endpoint_identity_options=0,
                  skipped_endpoint_identity_groups=set())
    original = _search_pair_geometry(
        structure_path=structure_path, binder_chain=binder_chain,
        binder_sequence=binder_sequence, mutable_positions=mutable_positions,
        library_path=library_path, reference_sequence=reference_sequence,
        max_total_mutations=max_total_mutations, max_edits=max_edits,
        sites=sites, parameters=parameters, excluded_aas=excluded_aas,
        _pair_set=search)
    options = search['options']
    # Top-level provenance uses the same native schema; chosen pair snapshots
    # below remain complete and unchanged, with their pair-only edits.
    metadata = copy.deepcopy(original['metadata'])
    metadata.update(
        pair_count=pair_count, pairs=[], candidate_cap=search['candidate_cap'],
        validated_candidate_options=len(options),
        search_exhausted=search['search_exhausted'],
        candidate_search_exhausted=search['search_exhausted'],
        candidate_cap_reached=search['candidate_cap_reached'],
        endpoint_identity_option_cap=search['endpoint_identity_option_cap'],
        retained_endpoint_identity_groups=len(search['endpoint_identity_counts']),
        endpoint_identity_cap_reached=bool(search['skipped_endpoint_identity_options']),
        skipped_endpoint_identity_options=search['skipped_endpoint_identity_options'],
        skipped_endpoint_identity_groups=len(search['skipped_endpoint_identity_groups']),
        skipped_options_interpretation='option combinations skipped before geometry validation',
        combination_search_cap=100000, combination_search_steps=0,
        combination_search_exhausted=False, combination_cap_reached=False,
        search_algorithm='ranked options with endpoint/identity diversity cap; lexicographic disjoint-pair backtracking',
        cross_pair_clash_check='all selected heavy sidechains against each other and original environment',
        sites=[], placed_sidechains=[], direct_pair_hbonds=[], source=None,
        edits=[], total_mutations=sum(a != b for a, b in
                                      zip(reference_sequence, binder_sequence)))
    # The collector's final result can contain an individual proposal; never
    # expose its single-pair selection as an accepted aggregate on failure.
    for key in ('library_index', 'library_roles', 'mode', 'backbone_rmsd_A',
                'pair_transform', 'fixed_site_identities_unchanged'):
        metadata.pop(key, None)
    endpoints, edits, geometries = [], [], []
    for option in options:
        pair_meta = option['metadata']
        keys = tuple((s['chain'], s['position_zero_based']) for s in pair_meta['sites'])
        endpoints.append(frozenset(keys))
        edits.append(tuple((e['position_zero_based'], e['to']) for e in pair_meta['edits']))
        geometries.append((keys, tuple(
            dict(coords={name: _xyz(xyz) for name, xyz in placed['coordinates'].items()})
            for placed in pair_meta['placed_sidechains'])))
    scale = metadata['parameters']['clash_radius_scale']

    def combine(start, selected, used, sequence, edit_count, keys, placed):
        remaining = pair_count-len(selected)
        if not remaining:
            return selected, sequence
        if len(options)-start < remaining:
            return None
        for index in range(start, len(options)-remaining+1):
            if metadata['combination_search_steps'] >= metadata['combination_search_cap']:
                metadata['combination_cap_reached'] = True
                return None
            metadata['combination_search_steps'] += 1
            if used & endpoints[index]:
                continue
            count = edit_count+len(edits[index])
            # Disjoint endpoints make edit counts additive; future pairs may
            # all be unchanged, so no positive edit lower bound is valid.
            if count > max_edits:
                continue
            proposed = list(sequence)
            for position, aa in edits[index]:
                proposed[position] = aa
            proposed = ''.join(proposed)
            total = sum(a != b for a, b in zip(reference_sequence, proposed))
            recoverable = min(max_edits-count, 2*(remaining-1))
            if total > max_total_mutations+recoverable:
                continue
            next_keys = keys+geometries[index][0]
            next_placed = placed+geometries[index][1]
            if _clash(search['environment'], next_keys, next_placed, scale):
                continue
            match = combine(index+1, selected+(index,), used | endpoints[index],
                            proposed, count, next_keys, next_placed)
            if match is not None:
                return match
            if metadata['combination_cap_reached']:
                return None
        return None

    match = combine(0, (), frozenset(), binder_sequence, 0, (), ())
    metadata['search_incomplete'] = (not metadata['candidate_search_exhausted'] or
                                     metadata['combination_cap_reached'])
    metadata['search_exhausted'] = not metadata['search_incomplete']
    if match is None:
        metadata['combination_search_exhausted'] = not metadata['combination_cap_reached']
        metadata['failure_reason'] = (
            'combination_cap_reached' if metadata['combination_cap_reached'] else
            'candidate_cap_reached_without_compatible_set' if metadata['candidate_cap_reached'] else
            'endpoint_identity_cap_reached_without_compatible_set' if metadata['endpoint_identity_cap_reached'] else
            'no_compatible_pair_set')
        return dict(status='insufficient_pairs', designed_sequence=binder_sequence,
                    metadata=metadata)
    selected, designed = match
    metadata['pairs'] = [options[i]['metadata'] for i in selected]
    metadata['selected_candidate_indices'] = list(selected)
    for pair_index, pair_meta in enumerate(metadata['pairs']):
        # Flattened geometry also supports the existing explicit preview helper.
        # Pair-local site indices stay intact inside metadata['pairs'].
        for record in pair_meta['sites']:
            metadata['sites'].append(dict(record, site_index=len(metadata['sites']),
                                          pair_index=pair_index))
        for record in pair_meta['placed_sidechains']:
            metadata['placed_sidechains'].append(
                dict(record, site_index=len(metadata['placed_sidechains']), pair_index=pair_index))
        for record in pair_meta['direct_pair_hbonds']:
            metadata['direct_pair_hbonds'].append(dict(
                record, donor_site_index=2*pair_index+record['donor_site_index'],
                acceptor_site_index=2*pair_index+record['acceptor_site_index'],
                pair_index=pair_index))
    metadata['edits'] = [dict(position_zero_based=p, position_one_based=p+1,
                              **{'from': a, 'to': b})
                         for p, (a, b) in enumerate(zip(binder_sequence, designed)) if a != b]
    metadata['total_mutations'] = sum(a != b for a, b in zip(reference_sequence, designed))
    metadata['fixed_site_identities_unchanged'] = all(
        pair['fixed_site_identities_unchanged'] for pair in metadata['pairs'])
    return dict(status='proposed', designed_sequence=designed, metadata=metadata)


def build_preview_model(structure_path, result):
    """Build complete heavy-sidechain pose from a proposed result; no file writes.

    Backbone and every uninvolved residue/chain retain input coordinates. Fixed
    target sidechains are not touched. Stale/mismatched input files are rejected.
    """
    if result.get('status') != 'proposed':
        raise ValueError('a proposed result is required for a preview')
    meta = result['metadata']
    if hashlib.sha256(Path(structure_path).read_bytes()).hexdigest() != meta['structure_sha256']:
        raise ValueError('preview structure differs from search input')
    model, _, observed = read_structure(structure_path, meta['binder_chain'])
    if observed != meta['original_binder_sequence']:
        raise ValueError('preview binder sequence mismatch')
    model = copy.deepcopy(model)
    chains = {c.id: [r for r in c if r.id[0] == ' '] for c in model}
    for placed in meta['placed_sidechains']:
        if placed['sidechain_policy'] == 'fixed':
            continue
        residue = chains[placed['chain']][placed['position_zero_based']]
        for atom in list(residue):
            if atom.name not in MAINCHAIN:
                residue.detach_child(atom.id)
        residue.resname = seq3(placed['residue_type']).upper()
        for name, xyz in placed['coordinates'].items():
            residue.add(Atom(name, _xyz(xyz), 0.0, 1.0, ' ', f'{name:>4}', 0, element=name[0]))
    sequences = {c: ''.join(seq1(r.resname, custom_map={'UNK': 'X'}) for r in rr) for c, rr in chains.items()}
    if sequences[meta['binder_chain']] != result['designed_sequence']:
        raise ValueError('preview/result binder sequence mismatch')
    if any(sequences[c] != seq for c, seq in meta['target_sequences_before'].items()):
        raise ValueError('preview changed a target sequence')
    return model


def write_preview(structure_path, result, output_path):
    """Explicitly export a complete PDB/mmCIF pose to a caller-selected path."""
    model = build_preview_model(structure_path, result)
    output_path = Path(output_path)
    if output_path.resolve() == Path(structure_path).resolve():
        raise ValueError('preview must not overwrite its input structure')
    writer = MMCIFIO() if output_path.suffix.lower() in ('.cif', '.mmcif') else PDBIO()
    writer.set_structure(model)
    writer.save(str(output_path))
    return str(output_path)
