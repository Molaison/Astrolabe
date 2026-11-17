"""Map positional selections to score observations without cropping their context."""
from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import copy
import re
import numpy as np
from astrolabe.core import global_config
from astrolabe.core.chain_resolution import (
    chain_ids_for_oligo, chain_to_protomer, resolve_selector_chain_ids,
)
from astrolabe.loss_config import validate_loss_selector
from astrolabe.selectors.utils import construct_position_selector


def _chain_tokens(value, *, parameter):
    if isinstance(value, str):
        tokens = value.split(",")
    elif isinstance(value, Sequence):
        tokens = list(value)
    else:
        raise ValueError(f"{parameter} must be a chain selector or list of selectors")
    if not tokens or any(not isinstance(token, str) or not token.strip() for token in tokens):
        raise ValueError(f"{parameter} must contain nonempty chain selectors")
    return [token.strip() for token in tokens]


def resolve_loss_chain_groups(
    oligo_name, protomer_a=None, protomer_b=None, included_protomers=None,
):
    """Resolve copies or protomer groups, preserving chain-first semantics."""
    if included_protomers is not None:
        legacy = _chain_tokens(included_protomers, parameter="included_protomers")
        if len(legacy) > 2:
            raise ValueError("included_protomers accepts at most two groups; use protomer_a/protomer_b")
        if protomer_a is not None or protomer_b is not None:
            raise ValueError("included_protomers conflicts with protomer_a/protomer_b; use one form")
        protomer_a = legacy[0]
        if len(legacy) == 2:
            protomer_b = legacy[1]
    ordered = chain_ids_for_oligo(oligo_name)
    def resolve(value, default, parameter):
        if value is None:
            return list(default)
        result = []
        for token in _chain_tokens(value, parameter=parameter):
            found = resolve_selector_chain_ids(oligo_name, token)
            if not found:
                raise ValueError(f"{parameter} selector {token!r} matches no chain in {oligo_name!r}")
            result.extend(chain for chain in found if chain not in result)
        return result
    group_a = resolve(protomer_a, ordered[:1], "protomer_a")
    group_b = resolve(protomer_b, [chain for chain in ordered if chain not in group_a], "protomer_b")
    overlap = set(group_a) & set(group_b)
    if overlap:
        raise ValueError(f"Scoring chain groups must be disjoint; overlap={sorted(overlap)!r}")
    return group_a, group_b


def _oligomer_view(eval_result, oligo_name):
    if hasattr(eval_result, "filter_by_oligomer"):
        result = eval_result.filter_by_oligomer([oligo_name])
    else:
        result = copy(eval_result)
        result.names = [oligo_name]
        for name, value in vars(eval_result).items():
            if isinstance(value, Mapping) and oligo_name in value:
                setattr(result, name, {oligo_name: value[oligo_name]})
    poses = getattr(result, "complex_poses", None)
    if poses is not None:
        result.complex_poses = copy(poses)
        result.complex_poses.poses = {
            name: pose for name, pose in (poses.poses or {}).items() if name == oligo_name
        }
        paths = getattr(result, "complex_paths", None)
        if paths:
            result.complex_poses.complex_paths = dict(paths)
    return result


def _range_ordinals(value, length, chain):
    if value is None:
        return frozenset(range(1, length + 1))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Residue ranges must be a nonempty string; omit them to select all")
    selected = set()
    for token in value.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", token)
        if match is None:
            raise ValueError(f"Invalid chain-relative residue range {token!r}")
        start, end = int(match[1]), int(match[2] or match[1])
        if start < 1 or end < start or end > length:
            raise ValueError(f"Residue range {token!r} exceeds chain {chain!r} length {length}")
        selected.update(range(start, end + 1))
    return frozenset(selected)


def selected_residue_ordinals(
    eval_result, oligo_name, chains, *, chain_lengths, selector=None, residues=None,
):
    """Return per-chain 1-based sequence ordinals; empty masks stay empty.

    Expressions use the existing selector registry and shared-protomer semantics.
    Chain selectors choose copies first. Legacy residue ranges intersect the
    positional selection. PDB residue numbers are not sequence positions.
    """
    chains = list(chains)
    unknown = set(chains) - set(chain_lengths)
    if unknown:
        raise ValueError(f"Selected chains are absent from the scored structure: {sorted(unknown)}")
    validate_loss_selector(selector)
    selector = selector.strip() if isinstance(selector, str) else selector
    selections = {chain: _range_ordinals(residues, chain_lengths[chain], chain) for chain in chains}
    if selector in (None, "None", "True"):
        return selections
    if selector == "False":
        return {chain: frozenset() for chain in chains}
    protomers = getattr(global_config, "protomers", None) or {}
    chain_map = chain_to_protomer(oligo_name)
    required = {chain_map[chain] for chain in chains if chain in chain_map}
    if any(chain not in chain_map for chain in chains) or required - set(protomers):
        raise ValueError(f"Positional selector has no current protomer mapping for {oligo_name!r}")
    selected = construct_position_selector(
        selector, _oligomer_view(eval_result, oligo_name),
        {name: protomers[name] for name in set(oligo_name) if name in protomers},
        strict=True,
    )
    for chain in chains:
        proto = chain_map[chain]
        mask = np.asarray(selected[proto], dtype=bool)
        expected = chain_lengths[chain]
        if mask.shape != (expected,):
            raise ValueError(
                f"Selector for {oligo_name}:{chain} ({proto}) has shape {mask.shape}; "
                f"scored structure has {expected} residues"
            )
        selections[chain] &= frozenset(int(index) + 1 for index in np.flatnonzero(mask))
    return selections


def loss_atom_mask(eval_result, oligo_name, atoms, chains, *, selector=None, residues=None):
    """Return an atom mask over a full AtomArray, preserving insertion codes."""
    chain_ids = np.asarray(atoms.chain_id).astype(str)
    residue_ids = np.asarray(atoms.res_id)
    insertions = np.asarray(getattr(atoms, "ins_code", np.full(len(chain_ids), ""))).astype(str)
    if chain_ids.ndim != 1 or residue_ids.shape != chain_ids.shape or insertions.shape != chain_ids.shape:
        raise ValueError("Scored atom residue annotations must be aligned one-dimensional arrays")
    positions = np.empty(len(chain_ids), dtype=int)
    identifiers = defaultdict(dict)
    for index, (chain, residue, insertion) in enumerate(zip(chain_ids, residue_ids, insertions)):
        key = (int(residue), str(insertion))
        positions[index] = identifiers[chain].setdefault(key, len(identifiers[chain]) + 1)
    selections = selected_residue_ordinals(
        eval_result, oligo_name, chains,
        chain_lengths={chain: len(values) for chain, values in identifiers.items()},
        selector=selector, residues=residues,
    )
    return np.fromiter(
        (position in selections.get(chain, ()) for chain, position in zip(chain_ids, positions)),
        dtype=bool, count=len(chain_ids),
    )


def pose_residue_indices(pose, selections):
    """Project per-chain sequence ordinals onto 1-based Pose residue indices."""
    counters = defaultdict(int)
    selected = []
    pdb = pose.pdb_info()
    if pdb is None:
        raise ValueError("Scored Pose must retain its chain identifiers")
    for index in range(1, pose.total_residue() + 1):
        chain = str(pdb.chain(index))
        counters[chain] += 1
        if counters[chain] in selections.get(chain, ()):
            selected.append(index)
    for chain, ordinals in selections.items():
        if chain not in counters or any(position < 1 or position > counters[chain] for position in ordinals):
            raise ValueError(f"Selected residue positions do not match Pose chain {chain!r}")
    return selected
