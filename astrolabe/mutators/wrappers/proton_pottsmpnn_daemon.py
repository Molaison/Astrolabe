#!/usr/bin/env python3
"""Long-lived Proton-PottsMPNN inference worker using JSONL over stdin/stdout."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import operator
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def _timing(phase: str, started: float, request_id=None) -> None:
    if os.environ.get("GH_PROTON_TIMING") == "1":
        print(json.dumps({"event": "proton_timing", "phase": phase,
                          "duration_ms": (time.perf_counter() - started) * 1000,
                          "epoch": time.time(), "pid": os.getpid(),
                          "request_id": request_id}), file=sys.stderr, flush=True)


V6_DEP_MAP = {
    "HIS-P": ["HIS-S"],
    "ASP-P": ["ASP-D"],
    "GLU-P": ["GLU-D"],
}
V3_V4_NEUTRAL_HIS_TOKENS = frozenset({"HID", "HIE", "HIS-D"})
DEFAULT_FORBIDDEN_TOKENS = ("HIS-A", "ASP-A", "GLU-A", "HIS-D")


THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

# Canonical parents plus the nine native v6 microstates; UNK is not a genotype.
V6_BINDER_TOKENS = frozenset(THREE_TO_ONE) | frozenset({
    "HIS-P", "HIS-S", "HIS-A", "ASP-P", "ASP-D", "ASP-A",
    "GLU-P", "GLU-D", "GLU-A",
})


def _validated_encoded_binder_tokens(value, binder_sequence):
    if value is None:
        return None, binder_sequence
    if not isinstance(value, list) or len(value) != len(binder_sequence):
        raise ValueError("encoded_binder_tokens must be a full binder-length list of v6 tokens.")
    if any(not isinstance(token, str) or token not in V6_BINDER_TOKENS for token in value):
        raise ValueError("encoded_binder_tokens contains an invalid v6 canonical/microstate token.")
    return list(value), "".join(THREE_TO_ONE[token.split("-")[0]] for token in value)


def _validated_state_patterns(value, n_centers):
    """Normalize explicit names/patterns; None preserves the unbounded legacy pair."""
    if value is None:
        return None
    if n_centers > 4:
        raise ValueError("state_patterns audit supports at most 4 His centers.")
    if not isinstance(value, (list, dict)) or not 1 <= len(value) <= 16:
        raise ValueError("state_patterns must be a nonempty list or named mapping, at most 16 patterns.")
    named = isinstance(value, dict)
    entries = value.items() if named else ((None, pattern) for pattern in value)
    patterns = []
    seen = set()
    for name, raw in entries:
        if isinstance(raw, str):
            pattern = raw
        elif isinstance(raw, list) and all(isinstance(token, str) and token in {"S", "P"} for token in raw):
            pattern = "".join(raw)
        else:
            raise ValueError("Each state_patterns entry must be an S/P string or list of S/P tokens.")
        if len(pattern) != n_centers or any(token not in "SP" for token in pattern):
            raise ValueError("state_patterns length must match histidine_positions and contain only S/P.")
        if pattern in seen:
            raise ValueError("state_patterns must not contain duplicate patterns.")
        if named and (not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name) is None):
            raise ValueError("state_patterns names must be 1-64 ASCII letters/digits/underscore/dot/hyphen.")
        seen.add(pattern)
        patterns.append({"name": name if named else pattern, "pattern": pattern})
    return patterns


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proton-PottsMPNN JSONL daemon")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hbplus-bin", required=True)
    parser.add_argument("--extended-vocab", default="v6")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--design-mask-adapter", choices=("none", "preserve"), default="none")
    return parser.parse_args()


def _write(stream, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
    stream.flush()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_runtime(args: argparse.Namespace):
    repo_path = Path(args.repo_path).resolve()
    sys.path.insert(0, str(repo_path / "foundry" / "src"))
    sys.path.insert(0, str(repo_path / "foundry" / "models" / "mpnn" / "src"))
    os.environ["HBPLUS_PATH"] = str(Path(args.hbplus_bin).resolve())

    started = time.perf_counter()
    import torch  # noqa: F401
    _timing("import_torch", started)

    started = time.perf_counter()
    from mpnn.inference_engines.potts_mpnn_ph import (
        PHDesignCriteria,
        PottsMPNNPHEngine,
    )
    _timing("import_engine", started)

    if getattr(args, "design_mask_adapter", "none") == "preserve":
        # The daemon's script directory is on sys.path in the isolated Python.
        from proton_design_mask_adapter import install_design_mask_adapter
        install_design_mask_adapter()

    device = None if args.device == "auto" else args.device
    started = time.perf_counter()
    engine = PottsMPNNPHEngine(
        checkpoint_path=str(Path(args.checkpoint).resolve()),
        extended_vocab=args.extended_vocab,
        device=device,
        out_directory=str(Path(args.output_dir).resolve()),
        write_fasta=False,
        write_structures=False,
    )
    _timing("construct_engine", started)
    return engine, PHDesignCriteria


def _load_atom_array(pdb_path: Path):
    from atomworks.io import parse
    from atomworks.io.parser import STANDARD_PARSER_ARGS

    data = parse(
        filename=str(pdb_path),
        keep_cif_block=True,
        **dict(STANDARD_PARSER_ARGS),
    )
    if "assemblies" in data and data["assemblies"]:
        return data["assemblies"]["1"][0]
    return data["asym_unit"][0]


def _token_chain_sequences(atom_array) -> tuple[Any, dict[str, str]]:
    from atomworks.ml.utils.token import get_token_starts

    token_aa = atom_array[get_token_starts(atom_array)]
    observed: dict[str, list[str]] = {}
    for index in range(len(token_aa)):
        chain = str(token_aa.chain_id[index])
        res_name = str(token_aa.res_name[index]).upper()
        aa = THREE_TO_ONE.get(res_name)
        if aa is None:
            raise ValueError(
                "Proton-PottsMPNN requires standard protein residues; "
                f"chain {chain!r} residue {token_aa.res_id[index]} has res_name {res_name!r}."
            )
        observed.setdefault(chain, []).append(aa)
    return token_aa, {chain: "".join(seq) for chain, seq in observed.items()}


def _validate_chain_sequences(
    observed: dict[str, str], expected: dict[str, str]
) -> None:
    for chain, expected_seq in expected.items():
        chain = str(chain)
        if chain not in observed:
            raise ValueError(
                f"Proton-PottsMPNN structure is missing expected chain {chain!r}."
            )
        observed_seq = observed[chain]
        if observed_seq == expected_seq:
            continue
        limit = min(len(observed_seq), len(expected_seq))
        first_diff = next(
            (i for i in range(limit) if observed_seq[i] != expected_seq[i]),
            limit,
        )
        raise ValueError(
            f"Proton-PottsMPNN chain {chain!r} sequence differs from the GH protomer "
            f"sequence at 0-based position {first_diff}: "
            f"structure={observed_seq[first_diff:first_diff + 12]!r} "
            f"expected={expected_seq[first_diff:first_diff + 12]!r} "
            f"(lengths {len(observed_seq)} vs {len(expected_seq)})."
        )


def _canonical_sequence(value: Any, label: str) -> str:
    sequence = str(value or "").strip().upper()
    invalid = sorted(set(sequence) - CANONICAL_AA)
    if not sequence:
        raise ValueError(f"Proton-PottsMPNN {label} must be nonempty.")
    if invalid:
        raise ValueError(
            f"Proton-PottsMPNN {label} must contain canonical amino acids only; "
            f"invalid={invalid}."
        )
    return sequence


def _histidine_positions(
    binder_sequence: str, value: Any
) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "Proton-PottsMPNN score_states histidine_positions must be a list."
        )
    positions = []
    for raw in value:
        if isinstance(raw, bool):
            raise ValueError(
                "Proton-PottsMPNN score_states histidine_positions cannot contain booleans."
            )
        try:
            position = operator.index(raw)
        except TypeError as exc:
            raise ValueError(
                "Proton-PottsMPNN score_states histidine_positions must contain integers."
            ) from exc
        if str(raw).strip() != str(position):
            raise ValueError(
                "Proton-PottsMPNN score_states histidine_positions must contain integers."
            )
        if position < 0 or position >= len(binder_sequence):
            raise ValueError(
                "Proton-PottsMPNN score_states histidine position "
                f"{position} is outside binder length {len(binder_sequence)}."
            )
        if binder_sequence[position] != "H":
            raise ValueError(
                "Proton-PottsMPNN score_states histidine position "
                f"{position} is {binder_sequence[position]!r}, not His."
            )
        positions.append(position)
    if not positions:
        raise ValueError(
            "Proton-PottsMPNN score_states histidine_positions must be nonempty."
        )
    if len(positions) != len(set(positions)):
        raise ValueError(
            "Proton-PottsMPNN score_states histidine_positions must be unique."
        )
    return positions


def _binder_token_mask(
    chain_ids: list[str], binder_chain: str, fixed_positions: list[int]
) -> list[bool]:
    fixed = {int(p) for p in fixed_positions}
    mask = [False] * len(chain_ids)
    binder_index = 0
    for token_index, chain in enumerate(chain_ids):
        if str(chain) != binder_chain:
            continue
        mask[token_index] = binder_index not in fixed
        binder_index += 1
    if binder_index == 0:
        raise ValueError(
            f"Proton-PottsMPNN binder chain {binder_chain!r} has no residues in the structure."
        )
    if any(p < 0 or p >= binder_index for p in fixed):
        raise ValueError(
            "Proton-PottsMPNN fixed_positions must be 0-based and within the binder "
            f"chain length {binder_index}."
        )
    return mask


def _apply_design_mask(atom_array, binder_chain: str, fixed_positions: list[int]) -> None:
    import numpy as np
    from atomworks.ml.utils.token import get_token_starts, spread_token_wise

    token_starts = get_token_starts(atom_array)
    token_chain_ids = [str(c) for c in atom_array.chain_id[token_starts]]
    token_mask = np.asarray(
        _binder_token_mask(token_chain_ids, binder_chain, fixed_positions),
        dtype=bool,
    )
    atom_mask = spread_token_wise(atom_array, token_mask)
    atom_array.set_annotation(
        "mpnn_designed_residue_mask", np.asarray(atom_mask, dtype=bool)
    )


def _design_metadata(design) -> dict[str, Any]:
    metadata = dict(design.to_metadata())
    metadata["canonical_sequence"] = str(design.canonical_sequence)
    metadata["extended_tokens"] = list(design.extended_tokens)
    trajectory = getattr(design, "energy_trajectory", None)
    if trajectory is not None:
        metadata["energy_trajectory"] = trajectory
    return metadata


def _token_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bytes):
        value = [value.decode("utf-8")]
    elif isinstance(value, str):
        value = [value]
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(
            f"Proton-PottsMPNN {label} must be a list of tokens."
        ) from exc
    return [str(item).strip() for item in items if str(item).strip()]


def _validate_criteria(criteria_cls, criteria: dict[str, Any], max_mutations: int):
    known = set(criteria_cls.__dataclass_fields__)
    unknown = sorted(set(criteria) - known)
    if unknown:
        raise ValueError(
            "Proton-PottsMPNN criteria contains unknown field(s): "
            + ", ".join(unknown)
        )
    criteria = dict(criteria)
    seed_source = str(criteria.get("seed_source") or "inverse").strip().lower()
    method = str(criteria.get("method") or "converged_mcmc").strip().lower()
    infill_scope = str(criteria.get("infill_scope") or "neighbourhood").strip().lower()
    try:
        center_count = int(criteria.get("center_count", 1))
    except (TypeError, ValueError):
        center_count = 1
    centre_free = (
        method == "greedy_energy_block"
        and infill_scope == "chain"
        and center_count == 0
        and not criteria.get("center_types")
        and not criteria.get("explicit_centers")
    )
    if seed_source not in {"inverse", "native"}:
        raise ValueError(
            "Proton-PottsMPNN criteria.seed_source must be 'inverse' or 'native'."
        )
    if seed_source == "native" and not centre_free:
        raise ValueError(
            "Proton-PottsMPNN criteria.seed_source='native' is supported only for "
            "centre-free method='greedy_energy_block' with infill_scope='chain', "
            "center_count=0, and empty center_types/explicit_centers; placement "
            "mode requires seed_source='inverse'."
        )
    criteria["seed_source"] = seed_source
    dep_map = criteria.get("dep_map")
    if dep_map is None:
        dep_map = V6_DEP_MAP
    elif not isinstance(dep_map, dict):
        raise ValueError("Proton-PottsMPNN criteria.dep_map must be a mapping.")
    merged_dep_map = {key: list(values) for key, values in V6_DEP_MAP.items()}
    for key, values in dep_map.items():
        if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
            raise ValueError(
                "Proton-PottsMPNN criteria.dep_map values must be lists of tokens."
            )
        merged_dep_map[str(key)] = [str(token) for token in values]
    invalid_his_tokens = sorted(
        {
            token
            for values in merged_dep_map.values()
            for token in values
            if token in V3_V4_NEUTRAL_HIS_TOKENS
        }
    )
    if invalid_his_tokens:
        raise ValueError(
            "Proton-PottsMPNN uses the v6 vocabulary; neutral histidine is HIS-S, "
            f"not {invalid_his_tokens}. Remove v3/v4 HID/HIE mappings."
        )
    criteria["dep_map"] = merged_dep_map

    forbidden = list(DEFAULT_FORBIDDEN_TOKENS)
    for token in _token_list(criteria.get("forbidden_tokens"), "forbidden_tokens"):
        if token not in forbidden:
            forbidden.append(token)
    criteria["forbidden_tokens"] = forbidden
    center_type_conflicts = [
        token
        for token in _token_list(criteria.get("center_types"), "center_types")
        if token in forbidden
    ]
    if center_type_conflicts:
        raise ValueError(
            "Proton-PottsMPNN forbidden_tokens conflicts with center_types "
            f"{center_type_conflicts}."
        )

    configured_max = criteria.get("max_mutations", None)
    if configured_max in (None, "", "None", "null", 0):
        criteria["max_mutations"] = int(max_mutations)
    else:
        configured_max = int(configured_max)
        if configured_max <= 0:
            raise ValueError("Proton-PottsMPNN criteria.max_mutations must be positive.")
        if configured_max > int(max_mutations):
            raise ValueError(
                "Proton-PottsMPNN criteria.max_mutations "
                f"({configured_max}) exceeds the resolved designable cap "
                f"({max_mutations}); increase mutation_num or lower the engine cap."
            )
        criteria["max_mutations"] = configured_max
    return criteria


def _validate_explicit_centers(
    criteria: dict[str, Any], token_aa, binder_chain: str, fixed_positions: list[int]
) -> None:
    centers = criteria.get("explicit_centers") or []
    if not centers:
        return
    binder_positions = [
        index
        for index, chain in enumerate(token_aa.chain_id)
        if str(chain) == binder_chain
    ]
    position_by_res_id = {
        int(token_aa.res_id[index]): binder_index
        for binder_index, index in enumerate(binder_positions)
    }
    fixed = {int(position) for position in fixed_positions}
    forbidden = {str(token) for token in criteria.get("forbidden_tokens") or []}
    for center in centers:
        if not isinstance(center, dict) or "res_id" not in center:
            raise ValueError(
                "Proton-PottsMPNN explicit_centers entries require res_id and "
                "protonation_type."
            )
        protonation_type = str(center.get("protonation_type") or "").strip()
        if protonation_type and protonation_type in forbidden:
            raise ValueError(
                "Proton-PottsMPNN explicit centre "
                f"protonation_type {protonation_type!r} is forbidden by "
                "exclude_AA/forbidden_tokens."
            )
        res_id = int(center["res_id"])
        binder_index = position_by_res_id.get(res_id)
        if binder_index is None:
            raise ValueError(
                f"Proton-PottsMPNN explicit centre res_id {res_id!r} is not in "
                f"binder chain {binder_chain!r}."
            )
        if binder_index in fixed:
            raise ValueError(
                f"Proton-PottsMPNN explicit centre res_id {res_id!r} is a fixed "
                "position; fixed residues cannot be redesigned or protonated by "
                "the model."
            )


def _infer(runtime, message: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    engine, criteria_cls = runtime
    started = time.perf_counter()
    pdb_path = Path(message["pdb_path"]).resolve()
    atom_array = _load_atom_array(pdb_path)
    token_aa, observed = _token_chain_sequences(atom_array)

    binder_chain = str(message["binder_chain"])
    binder_sequence = str(message["binder_sequence"]).strip().upper()
    chain_sequences = {
        str(k): str(v).strip().upper()
        for k, v in dict(message.get("chain_sequences") or {}).items()
    }
    _validate_chain_sequences(observed, chain_sequences)
    if observed.get(binder_chain) != binder_sequence:
        raise ValueError(
            f"Proton-PottsMPNN binder_sequence does not match chain {binder_chain!r} "
            "in the structure."
        )

    fixed_positions = [int(p) for p in message.get("fixed_positions") or []]
    _apply_design_mask(atom_array, binder_chain, fixed_positions)

    max_mutations = int(message.get("max_mutations") or 0)
    if max_mutations <= 0:
        raise ValueError("Proton-PottsMPNN max_mutations must be greater than zero.")
    criteria = _validate_criteria(
        criteria_cls, dict(message.get("criteria") or {}), max_mutations
    )
    _validate_explicit_centers(criteria, token_aa, binder_chain, fixed_positions)
    seed = int(message.get("seed", 0))
    n_jobs = max(1, int(message.get("n_jobs", 1) or 1))
    _timing("parse_validate_mask", started, message.get("id"))

    started = time.perf_counter()
    designs = engine.run_ph_redesign(
        atom_array=atom_array,
        binder_chain=binder_chain,
        criteria_list=[criteria_cls.from_params(criteria)],
        seed=seed,
        initial_sequences=[binder_sequence],
        n_jobs=n_jobs,
    )
    _timing("redesign", started, message.get("id"))
    started = time.perf_counter()
    # PHDesignSet carries the engine's per-seed energies as an attribute.  Convert
    # only after reading it: a plain list loses that attribute.
    seed_energies = getattr(designs, "seed_energies", None)
    designs = list(designs)
    if not designs:
        raise RuntimeError("Proton-PottsMPNN engine returned no designs.")
    selected = designs[0]
    if str(selected.canonical_sequence) != str(selected.canonical_sequence).strip().upper():
        raise RuntimeError("Proton-PottsMPNN selected sequence is not canonical uppercase.")
    if len(str(selected.canonical_sequence)) != len(binder_sequence):
        raise RuntimeError(
            "Proton-PottsMPNN selected sequence length does not match binder length."
        )

    request_id = str(message.get("id") or "request")
    raw_output_path = output_dir / f"{request_id}.json"
    raw_payload = {
        "type": "proton_pottsmpnn_raw",
        "request_id": request_id,
        "pdb_path": str(pdb_path),
        "binder_chain": binder_chain,
        "binder_sequence": binder_sequence,
        "fixed_positions": fixed_positions,
        "criteria": criteria,
        "seed": seed,
        "selected": _design_metadata(selected),
        "designs": [_design_metadata(design) for design in designs],
        "seed_energies": seed_energies,
    }
    _write_json_atomic(raw_output_path, raw_payload)
    _timing("serialize_write", started, request_id)
    return {
        "selected": _design_metadata(selected),
        "designs_considered": len(designs),
        "raw_output_path": str(raw_output_path),
        "seed_energies": seed_energies,
    }


def _score_states(
    runtime, message: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Score the legacy pair and optional mixed states on one fixed context."""
    engine, _criteria_cls = runtime
    started = time.perf_counter()
    pdb_path = Path(message["pdb_path"]).resolve()
    atom_array = _load_atom_array(pdb_path)
    token_aa, observed = _token_chain_sequences(atom_array)

    binder_chain = str(message["binder_chain"]).strip()
    binder_sequence = _canonical_sequence(
        message.get("binder_sequence"), "score_states binder_sequence"
    )
    chain_sequences = {
        str(chain_id): _canonical_sequence(
            sequence, f"score_states chain_sequences[{chain_id!r}]"
        )
        for chain_id, sequence in dict(message.get("chain_sequences") or {}).items()
    }
    if binder_chain not in chain_sequences:
        raise ValueError(
            "Proton-PottsMPNN score_states chain_sequences is missing binder "
            f"chain {binder_chain!r}."
        )
    if chain_sequences[binder_chain] != binder_sequence:
        raise ValueError(
            "Proton-PottsMPNN score_states binder_sequence does not match "
            f"chain_sequences[{binder_chain!r}]."
        )
    _validate_chain_sequences(observed, chain_sequences)
    encoded_tokens, scored_binder_sequence = _validated_encoded_binder_tokens(
        message.get("encoded_binder_tokens"), binder_sequence
    )
    positions = _histidine_positions(scored_binder_sequence, message.get("histidine_positions"))
    patterns = _validated_state_patterns(message.get("state_patterns"), len(positions))
    audit = patterns is not None or encoded_tokens is not None

    ctx = engine._build_context(atom_array, binder_chain)
    binder_positions = [
        int(position) for position in ctx.chainA_all_idx.tolist()
    ]
    if len(binder_positions) != len(binder_sequence):
        raise RuntimeError(
            "Proton-PottsMPNN score_states context binder length "
            f"{len(binder_positions)} != sequence length {len(binder_sequence)}."
        )
    if audit and (len(set(binder_positions)) != len(binder_positions)
            or len(ctx.S_native) != int(ctx.L)
            or any(position < 0 or position >= int(ctx.L) for position in binder_positions)):
        raise RuntimeError("Proton-PottsMPNN score_states invalid context binder mapping.")
    if audit:
        for local_position, context_position in enumerate(binder_positions):
            if (str(ctx.token_aa.chain_id[context_position]) != binder_chain
                    or THREE_TO_ONE.get(str(ctx.token_aa.res_name[context_position]).upper())
                    != binder_sequence[local_position]):
                raise RuntimeError("Proton-PottsMPNN score_states context differs from structure binder identity.")
    for local_position in positions:
        context_position = binder_positions[local_position]
        if encoded_tokens is None and str(ctx.token_aa.res_name[context_position]).upper() != "HIS":
            raise RuntimeError(
                "Proton-PottsMPNN score_states context does not map His "
                f"position {local_position} to HIS."
            )

    tokens = ctx.encoding.token_to_idx
    missing = [token for token in ("HIS-S", "HIS-P") if token not in tokens]
    if missing:
        raise KeyError(
            "Proton-PottsMPNN score_states requires v6 His tokens; missing "
            f"{missing}."
        )
    native_snapshot = [int(value) for value in ctx.S_native] if audit else None
    scored_native = ctx.S_native
    if encoded_tokens is not None:
        if any(token not in tokens for token in encoded_tokens):
            raise ValueError("encoded_binder_tokens contains tokens absent from the context v6 vocabulary.")
        scored_native = ctx.S_native.clone()
        for context_position, token in zip(binder_positions, encoded_tokens):
            scored_native[context_position] = tokens[token]
    scored_snapshot = [int(value) for value in scored_native] if audit else None
    neutral = scored_native.clone()
    protonated = scored_native.clone()
    context_positions = [binder_positions[position] for position in positions]
    for context_position in context_positions:
        neutral[context_position] = tokens["HIS-S"]
        protonated[context_position] = tokens["HIS-P"]
    selected = set(context_positions)
    outside_centers_unchanged = all(
        int(neutral[position]) == int(protonated[position])
        and (not audit or int(neutral[position]) == scored_snapshot[position])
        for position in range(int(ctx.L))
        if position not in selected
    )
    if not outside_centers_unchanged:
        raise RuntimeError(
            "Proton-PottsMPNN score_states changed a non-centre encoded state."
        )

    neutral_energy = float(ctx.scorer.H_of(neutral))
    protonated_energy = float(ctx.scorer.H_of(protonated))
    gap = protonated_energy - neutral_energy
    if not all(
        math.isfinite(value)
        for value in (neutral_energy, protonated_energy, gap)
    ):
        raise RuntimeError(
            "Proton-PottsMPNN score_states produced a non-finite energy."
        )

    audit_fields = {}
    if audit:
        idx_to_token = {int(index): token for token, index in tokens.items()}
        if len(idx_to_token) != len(tokens):
            raise RuntimeError("Proton-PottsMPNN score_states context vocabulary has duplicate indices.")

        def decode(values):
            try:
                return [idx_to_token[int(value)] for value in values]
            except KeyError as exc:
                raise RuntimeError("Proton-PottsMPNN score_states unknown context token index.") from exc

        requested = patterns if patterns is not None else [
            {"name": "S" * len(positions), "pattern": "S" * len(positions)},
            {"name": "P" * len(positions), "pattern": "P" * len(positions)},
        ]
        energy_cache = {"S" * len(positions): (neutral, neutral_energy),
                        "P" * len(positions): (protonated, protonated_energy)}
        states = []
        for spec in requested:
            pattern = spec["pattern"]
            if pattern not in energy_cache:
                encoded = scored_native.clone()
                for position, state in zip(context_positions, pattern):
                    encoded[position] = tokens[f"HIS-{state}"]
                energy_cache[pattern] = (encoded, float(ctx.scorer.H_of(encoded)))
            encoded, energy = energy_cache[pattern]
            if not math.isfinite(energy):
                raise RuntimeError("Proton-PottsMPNN score_states produced a non-finite energy.")
            states.append({**spec, "energy": energy,
                           "center_tokens": [f"HIS-{state}" for state in pattern],
                           "encoded_sequence": [int(value) for value in encoded],
                           "tokens": decode(encoded)})
        # Check every scored clone, including legacy endpoints absent from requested states.
        for pattern, (encoded, _) in energy_cache.items():
            expected = list(scored_snapshot)
            for position, state in zip(context_positions, pattern):
                expected[position] = tokens[f"HIS-{state}"]
            if [int(value) for value in encoded] != expected:
                raise RuntimeError("Proton-PottsMPNN score_states scorer changed encoded states.")
        if [int(value) for value in ctx.S_native] != native_snapshot:
            raise RuntimeError("Proton-PottsMPNN score_states scorer changed native context states.")
        audit_fields = {
            "state_names": [state["name"] for state in states],
            "states": states,
            "state_patterns": requested,
            "scored_binder_sequence": scored_binder_sequence,
            "context_binder_positions": binder_positions,
            "context_histidine_positions": context_positions,
            "scored_encoded_sequence": scored_snapshot,
            "scored_tokens": decode(scored_snapshot),
            "scored_binder_tokens": decode([scored_snapshot[p] for p in binder_positions]),
            "native_encoded_sequence": native_snapshot,
            "native_tokens": decode(native_snapshot),
            "token_to_idx": {token: int(index) for token, index in tokens.items()},
            "noncenter_context_positions": [p for p in range(int(ctx.L)) if p not in selected],
            "same_context": True,
            "audit_source": "fixed_geometry_potts_context",
            "energy_method": "ctx.scorer.H_of",
            "energy_semantics": "Potts energy only; no probability or pKa; no physical geometry validation",
        }

    request_id = str(message.get("id") or "request")
    raw_output_path = output_dir / f"score_states_{request_id}.json"
    raw_payload = {
        "type": "proton_pottsmpnn_state_score_raw",
        "request_id": request_id,
        "pdb_path": str(pdb_path),
        "binder_chain": binder_chain,
        "binder_sequence": binder_sequence,
        "chain_sequences": chain_sequences,
        "histidine_positions": positions,
        "neutral_token": "HIS-S",
        "protonated_token": "HIS-P",
        "neutral_energy": neutral_energy,
        "protonated_energy": protonated_energy,
        "gap": gap,
        "n_centers": len(positions),
        "outside_centers_unchanged": outside_centers_unchanged,
        **audit_fields,
    }
    if audit:
        raw_payload["input_request"] = message
        raw_payload["input_state_patterns"] = message.get("state_patterns")
        raw_payload["input_encoded_binder_tokens"] = message.get("encoded_binder_tokens")
        raw_payload["context_binder_positions"] = binder_positions
    _write_json_atomic(raw_output_path, raw_payload)
    _timing("score_states", started, request_id)
    return {
        "neutral_energy": neutral_energy,
        "protonated_energy": protonated_energy,
        "gap": gap,
        "n_centers": len(positions),
        "histidine_positions": positions,
        "binder_sequence": binder_sequence,
        "outside_centers_unchanged": outside_centers_unchanged,
        "raw_output_path": str(raw_output_path),
        **audit_fields,
    }


def main() -> int:
    args = _parse_args()
    protocol_out = sys.stdout
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(sys.stderr):
        started = time.perf_counter()
        runtime = _load_runtime(args)
        _timing("load_runtime", started)
    _write(
        protocol_out,
        {"type": "ready", "pid": os.getpid(), "model_load_count": 1},
    )

    request_count = 0
    for raw in sys.stdin:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            _write(
                protocol_out,
                {
                    "type": "error",
                    "id": None,
                    "error_type": "JSONDecodeError",
                    "error": str(exc),
                },
            )
            continue
        message_type = message.get("type")
        if message_type == "shutdown":
            _write(protocol_out, {"type": "ok", "message": "bye"})
            return 0
        if message_type not in {"infer", "score_states"}:
            _write(
                protocol_out,
                {
                    "type": "error",
                    "id": message.get("id"),
                    "error_type": "UnknownMessage",
                    "error": f"Unknown Proton-PottsMPNN message type: {message_type!r}",
                },
            )
            continue

        request_count += 1
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if message_type == "score_states":
                    result = _score_states(runtime, message, output_dir)
                else:
                    result = _infer(runtime, message, output_dir)
        except Exception as exc:  # noqa: BLE001 - reported to the parent process
            _write(
                protocol_out,
                {
                    "type": "error",
                    "id": message.get("id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            continue
        _write(
            protocol_out,
            {
                "type": "result",
                "id": message.get("id"),
                "pid": os.getpid(),
                "model_load_count": 1,
                "request_count": request_count,
                **result,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
