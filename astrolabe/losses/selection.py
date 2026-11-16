"""Resolve scoring regions without changing the structure or mutation masks.

Residue ranges are one-based ordinals within each selected chain, never PDB
residue numbers. Position-selector expressions use the existing protomer DSL;
the resulting protomer mask is expanded to every explicitly selected copy.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from astrolabe.core.chain_resolution import (
    chain_ids_for_oligo, chain_to_protomer, resolve_selector_chain_ids,
)


def _tokens(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = value.split(",")
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise ValueError("Expected a chain selector or a list of selectors") from exc
    if not values or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError("Chain selectors must be nonempty strings")
    return [v.strip() for v in values]


def resolve_chain_groups(oligo, protomer_a=None, protomer_b=None, included_protomers=None):
    """Resolve chain-first selectors, rejecting an ambiguous legacy alias."""
    if included_protomers is not None:
        alias = _tokens(included_protomers)
        if len(alias) != 1:
            raise ValueError("included_protomers accepts one selector; use protomer_a for a chain group")
        if protomer_a is not None and _tokens(protomer_a) != alias:
            raise ValueError("protomer_a and included_protomers disagree")
        protomer_a = alias

    ordered = chain_ids_for_oligo(oligo)

    def resolve(value, default):
        if value is None:
            return list(default)
        resolved = []
        for selector in _tokens(value):
            chains = resolve_selector_chain_ids(oligo, selector)
            if not chains:
                raise ValueError(f"Chain selector {selector!r} does not match oligomer {oligo!r}")
            resolved.extend(c for c in chains if c not in resolved)
        return resolved

    a = resolve(protomer_a, ordered[:1])
    b = resolve(protomer_b, [c for c in ordered if c not in a])
    if set(a) & set(b):
        raise ValueError("Interface chain groups must be disjoint")
    return a, b


def residue_positions(value, length: int) -> tuple[int, ...]:
    if value is None:
        return tuple(range(1, length + 1))
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, Sequence):
        parts = list(value)
    else:
        raise ValueError("Residues must be a one-based range string or list of positions")
    selected = set()
    for part in parts:
        if isinstance(part, bool):
            raise ValueError("Residue positions must be integers")
        token = str(part).strip()
        if not token:
            raise ValueError("An explicit residue selection must not be empty")
        pieces = token.split("-")
        if len(pieces) > 2 or not all(p.isdecimal() for p in pieces):
            raise ValueError(f"Invalid one-based residue range {token!r}")
        start, end = int(pieces[0]), int(pieces[-1])
        if start < 1 or end < start or end > length:
            raise ValueError(f"Residue range {token!r} is outside chain length {length}")
        selected.update(range(start, end + 1))
    if not selected:
        raise ValueError("An explicit residue selection must not be empty")
    return tuple(sorted(selected))


def validate_region_arguments(residues_a=None, residues_b=None, selector_a=None, selector_b=None):
    for side, residues, expression in (("a", residues_a, selector_a), ("b", residues_b, selector_b)):
        if residues is not None and expression is not None:
            raise ValueError(f"Use only one of protomer_{side}_residues and position_selector_{side}")
        if expression is not None and (not isinstance(expression, str) or not expression.strip()):
            raise ValueError(f"position_selector_{side} must be a nonempty selector expression")


@dataclass(frozen=True)
class InterfaceSelection:
    chains_a: tuple[str, ...]
    chains_b: tuple[str, ...]
    residues_a: tuple[int, ...]  # zero-based structure residue ordinals
    residues_b: tuple[int, ...]
    positions_a: Mapping[str, tuple[int, ...]]
    positions_b: Mapping[str, tuple[int, ...]]

    def evidence(self):
        return {"numbering": "one_based_chain_relative", "a": {c: list(p) for c, p in self.positions_a.items()},
                "b": {c: list(p) for c, p in self.positions_b.items()}}


def resolve_residue_selection(eval_result, oligo, residue_chains, chains_a, chains_b, *,
                              residues_a=None, residues_b=None, selector_a=None, selector_b=None):
    validate_region_arguments(residues_a, residues_b, selector_a, selector_b)
    by_chain = defaultdict(list)
    for index, chain in enumerate(residue_chains):
        by_chain[str(chain)].append(index)
    chain_map = chain_to_protomer(oligo)

    def select(chains, ranges, expression):
        masks = None
        if expression is not None:
            from astrolabe.core import global_config
            from astrolabe.selectors.utils import construct_position_selector
            protomers = global_config.protomers
            masks = construct_position_selector(expression, eval_result, protomers, missing="empty")
        indices, positions = [], {}
        for chain in chains:
            if chain not in by_chain:
                raise ValueError(f"Selected chain {chain!r} is missing from the scored structure")
            chain_indices = by_chain[chain]
            if masks is None:
                chosen = residue_positions(ranges, len(chain_indices))
            else:
                proto = chain_map[chain]
                if proto not in masks:
                    raise ValueError(f"Position selector has no mask for protomer {proto!r}")
                mask = np.asarray(masks[proto])
                if mask.shape != (len(chain_indices),) or mask.dtype != np.bool_:
                    raise ValueError(f"Position selector for chain {chain!r} must be a boolean mask of length {len(chain_indices)}")
                chosen = tuple(int(i) + 1 for i in np.flatnonzero(mask))
            positions[chain] = chosen
            indices.extend(chain_indices[p - 1] for p in chosen)
        if not indices:
            raise ValueError("Scoring region is empty; an empty positional selector never selects all residues")
        return tuple(indices), positions

    a, pos_a = select(chains_a, residues_a, selector_a)
    b, pos_b = select(chains_b, residues_b, selector_b)
    return InterfaceSelection(tuple(chains_a), tuple(chains_b), a, b, pos_a, pos_b)


def atom_residue_table(atoms):
    """Retain insertion codes and reject discontiguous duplicate identities."""
    size = len(atoms.coord)
    chains = np.asarray(atoms.chain_id)
    numbers = np.asarray(atoms.res_id)
    insertions = np.asarray(getattr(atoms, "ins_code", np.full(size, "")))
    if any(len(a) != size for a in (chains, numbers, insertions)):
        raise ValueError("Atom residue annotations do not align")
    keys, atom_residues, seen = [], [], set()
    previous = None
    for chain, number, insertion in zip(chains, numbers, insertions):
        key = (str(chain), int(number), str(insertion))
        if key != previous:
            if key in seen:
                raise ValueError(f"Discontiguous duplicate residue identity {key!r}")
            seen.add(key)
            keys.append(key)
            previous = key
        atom_residues.append(len(keys) - 1)
    return keys, np.asarray(atom_residues, dtype=np.intp)


def select_atoms(eval_result, oligo, atoms, chains_a, chains_b, **options):
    keys, owners = atom_residue_table(atoms)
    selected = resolve_residue_selection(eval_result, oligo, [k[0] for k in keys], chains_a, chains_b, **options)
    return selected, np.isin(owners, selected.residues_a), np.isin(owners, selected.residues_b)


def select_pose(eval_result, oligo, pose, chains_a, chains_b, **options):
    chains = [str(pose.pdb_info().chain(i)) for i in range(1, pose.total_residue() + 1)]
    return resolve_residue_selection(eval_result, oligo, chains, chains_a, chains_b, **options)
