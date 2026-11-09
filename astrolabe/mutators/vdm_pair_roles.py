"""Validate native vdM telemetry and expose its two physical endpoint roles.

This module only converts an existing proposal; it never queues mutations or
invokes a sampler. All position-keyed identity dictionaries use **0-based
polymer positions**, never PDB residue numbers. Their keys are Python integers
(JSON object serialization necessarily renders those keys as strings).
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path


_CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")
_SOURCE_IDENTITIES = {"HIS": "H", "LYS": "K", "ARG": "R", "ASP": "D", "GLU": "E"}


def _require(condition, message):
    if not condition:
        raise ValueError(f"vdm_pair_roles: {message}")


def _mapping(value, label):
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _items(value, label, *, size=None):
    _require(isinstance(value, (list, tuple)), f"{label} must be a list or tuple")
    _require(size is None or len(value) == size, f"{label} must contain {size} items")
    return value


def _integer(value, label, *, minimum=0):
    _require(not isinstance(value, bool) and isinstance(value, Integral),
             f"{label} must be an integer")
    value = int(value)
    _require(minimum is None or value >= minimum, f"{label} is out of range")
    return value


def _positions(value, length, label):
    positions = [_integer(p, label) for p in _items(value, label)]
    _require(len(set(positions)) == len(positions), f"{label} contains duplicate positions")
    _require(all(p < length for p in positions), f"{label} contains out-of-range positions")
    return set(positions)


def _sequence(value, label):
    _require(isinstance(value, str) and bool(value) and not set(value) - _CANONICAL,
             f"{label} must be a nonempty canonical uppercase sequence")
    return value


def _original_mask(proto, length):
    # The native property can lazily cache on self; isolate that cache write.
    mask = list(copy.copy(proto).mutable_mask)
    _require(len(mask) == length, "original protomer mutable-mask length mismatch")
    _require(all(type(m).__name__ in {"bool", "bool_"} for m in mask),
             "original protomer mutable_mask must be boolean")
    return [bool(m) for m in mask]


def _residue_id(value, *, native, label):
    values = _items(value, label, size=2 if native else 3)
    if not native:
        _require(isinstance(values[0], str), f"{label} has an invalid residue flag")
        values = values[1:]
    number = _integer(values[0], f"{label} PDB number", minimum=None)
    _require(isinstance(values[1], str), f"{label} has an invalid insertion code")
    return [number, values[1].strip()]


def _status(value, label):
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be nonempty")
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _is_noop(status):
    return status.startswith(("no_op", "noop", "no_match", "no_candidate", "no_pair"))


def _validate_edits(records, expected, numbering, *, native):
    label = "telemetry.edits" if native else "metadata.edits"
    observed = {}
    for raw in _items(records, label):
        record = _mapping(raw, label)
        p = _integer(record["position" if native else "position_zero_based"], label)
        _require(p not in observed and p < len(numbering), f"{label} has duplicate/invalid positions")
        if native:
            _require(_residue_id(record["residue_id"], native=True, label=label) == numbering[p],
                     f"{label} residue numbering disagrees with the structure")
        else:
            _require(_integer(record["position_one_based"], label, minimum=1) == p + 1,
                     f"{label} 1-based and 0-based positions disagree")
        observed[p] = (record["from"], record["to"])
    _require(observed == expected, f"{label} does not reconstruct designed_sequence from the pair")


def build_vdm_pair_roles(*, oligo_name, features, telemetry, protomers) -> dict | None:
    """Return ``vdm_pair_roles_v1`` for one native proposed pair, or ``None``.

    ``features`` and ``telemetry`` are the single-oligomer values produced by
    ``VdmPairMutator.prepare_mutation_input`` and ``prepare_mutation_output``.
    Invalid/stale/ambiguous inputs raise ``ValueError``. A no-match/no-op never
    yields independently runnable roles. Real protomers supply original
    sequences and masks; no global configuration is accessed by this converter.

    ``targets`` is keyed by physical label chain, preserving repeated target
    protomers. ``network_pair_roles`` is keyed by binder protomer and then by
    integer 0-based polymer position. Every target identity is fixed. Every
    selected His endpoint, including a His partner, requests neutral HIS-S as
    a design constraint, not a measured or inferred protonation assignment.
    Target His centers explicitly require backend support for that constraint.
    Pending target sequences must remain original; pending binder sequence must
    be either original (packet replay) or the complete designed pair sequence.
    """
    try:
        return _build(oligo_name, features, telemetry, protomers)
    except (KeyError, TypeError, AttributeError, IndexError, OSError, OverflowError) as exc:
        raise ValueError(f"vdm_pair_roles: invalid or incomplete native proposal: {exc}") from exc


def _build(oligo_name, features, telemetry, protomers):
    features = _mapping(features, "features")
    telemetry = _mapping(telemetry, "telemetry")
    status = _status(telemetry["status"], "telemetry.status")
    engine_status = _status(telemetry.get("engine_status", status), "telemetry.engine_status")
    if _is_noop(status):
        _require(_is_noop(engine_status) or engine_status == "proposed",
                 "no-op telemetry has an invalid engine status")
        original = features.get("sequence")
        for key in ("designed_sequence", "projected_sequence"):
            if key in telemetry:
                _require(telemetry[key] == original, f"no-op {key} changes the binder")
        _require(not telemetry.get("edits") and telemetry.get("n_edits", 0) == 0,
                 "no-op telemetry contains edits")
        return None
    _require(status == "proposed" and engine_status == "proposed", "unsupported/failure proposal status")
    _require(isinstance(oligo_name, str) and bool(oligo_name), "oligo_name must be nonempty")
    protomers = _mapping(protomers, "protomers")
    metadata = _mapping(telemetry["metadata"], "metadata")

    path = Path(features["structure_path"])
    _require(path.is_file(), f"original structure does not exist: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _require(metadata["structure_sha256"] == digest, "original structure SHA256 mismatch")
    # Import only for a genuine candidate. Reuse native parsing and label-chain
    # semantics, without constructing a mutator or reading its configuration.
    from astrolabe.mutators.vdm import _structure_sequences
    from astrolabe.utils.mmcif_ids import int_id_to_str_id

    try:
        sequences, numbering, author_map = _structure_sequences(path)
    except Exception as exc:
        raise ValueError(f"vdm_pair_roles: cannot parse original structure: {exc}") from exc
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
             "original structure changed during conversion")
    label_protomers = {int_id_to_str_id(i + 1): p for i, p in enumerate(oligo_name)}
    _require(set(sequences) == set(label_protomers), "oligo physical label chains disagree with structure")
    _require(set(author_map) == set(sequences)
             and all(isinstance(a, str) and bool(a) for a in author_map.values())
             and len(set(author_map.values())) == len(author_map),
             "NEED_DECISION: author-chain map must be one-to-one")
    expected_sequences = {}
    for label, name in label_protomers.items():
        _require(name in protomers, f"unknown protomer {name!r}")
        expected_sequences[label] = _sequence(str(protomers[name].sequence), f"protomer {name}")
    _require(sequences == expected_sequences, "complex sequences differ from original protomers")
    binder_label = features["native_chain"]
    _require(binder_label in sequences, "native binder label chain is absent")
    binder_author = author_map[binder_label]
    binder_proto = label_protomers[binder_label]
    _require(features["binder_chain"] == binder_author
             and features["binder_protomer"] == binder_proto
             and oligo_name.count(binder_proto) == 1,
             "binder label/author/protomer mapping is inconsistent or repeated")
    _require(telemetry["binder_chain"] == binder_label
             and telemetry["structure_chain"] == binder_author
             and metadata["binder_chain"] == binder_author, "telemetry binder-chain mapping disagrees")
    original = sequences[binder_label]
    _require(features["sequence"] == original and features["binder_sequence"] == original
             and metadata["original_binder_sequence"] == original, "stale original binder sequence")
    designed = _sequence(telemetry["designed_sequence"], "designed_sequence")
    _require(len(designed) == len(original), "designed binder length changed")
    _require(telemetry["projected_sequence"] == designed,
             "projected_sequence includes implicit concurrent mutations")
    pending_binder = _sequence(str(protomers[binder_proto].mutated_sequence),
                               "pending binder mutated_sequence")
    _require(pending_binder in (original, designed),
             "pending binder mutations are unrelated, mixed, or only a partial pair")
    for label, name in label_protomers.items():
        if label != binder_label:
            pending_target = _sequence(str(protomers[name].mutated_sequence),
                                      f"pending target {label} mutated_sequence")
            _require(pending_target == sequences[label],
                     f"pending target mutations change fixed physical chain {label}")
    binder_numbering = numbering[binder_label]
    feature_numbering = [_residue_id(r, native=True, label="features.residue_ids")
                         for r in _items(features["residue_ids"], "features.residue_ids")]
    _require(feature_numbering == binder_numbering, "features binder residue numbering is stale")

    # Read only: preserve even the original protomer's lazy mask-cache state.
    mask = _original_mask(protomers[binder_proto], len(original))
    mutable = _positions(features["mutable_positions"], len(original), "features.mutable_positions")
    _require(all(bool(mask[p]) for p in mutable), "features expands original seqs/mutable_mask constraints")
    effective = _positions(metadata["effective_mutable_positions"], len(original),
                           "metadata.effective_mutable_positions")
    _require(effective == mutable, "effective mutable positions differ from features")

    targets_before = {author_map[c]: seq for c, seq in sequences.items() if c != binder_label}
    _require(_mapping(metadata["target_sequences_before"], "target_sequences_before") == targets_before
             and _mapping(metadata["target_sequences_after"], "target_sequences_after") == targets_before,
             "all actual target chains must retain their original sequences")
    if "target_sequence_unchanged" in metadata:
        _require(metadata["target_sequence_unchanged"] is True, "target sequence change reported")

    raw_sites = _items(metadata["sites"], "metadata.sites", size=2)
    specs = _items(features["sites"], "features.sites", size=2)
    roles = _items(metadata["library_roles"], "metadata.library_roles", size=2)
    _require(all(isinstance(r, str) for r in roles) and set(roles) == {"his", "partner"},
             "library roles must be exactly his and partner")
    source = _mapping(metadata["source"], "metadata.source")
    author_labels = {a: c for c, a in author_map.items()}
    sites, pair_positions, pair_edits = [], {}, {}
    occupied = set()
    for i, raw in enumerate(raw_sites):
        raw = _mapping(raw, f"site {i}")
        spec = _mapping(specs[i], f"features site {i}")
        _require(_integer(raw["site_index"], "site_index") == i, "site_index disagrees with endpoint order")
        author = raw["chain"]
        _require(author in author_labels, "site author chain is absent from structure")
        label = author_labels[author]
        p = _integer(raw["position_zero_based"], "site position_zero_based")
        _require(p < len(sequences[label]) and (label, p) not in occupied,
                 "site is outside its physical chain or duplicates the other endpoint")
        occupied.add((label, p))
        _require(_integer(raw["position_one_based"], "site position_one_based", minimum=1) == p + 1,
                 "site 0-based/1-based positions disagree")
        rid = _residue_id(raw["residue_id"], native=False, label="site.residue_id")
        _require(rid == numbering[label][p], "site PDB number/insertion code disagrees with input structure")
        old, selected, role = raw["original_residue_type"], raw["residue_type"], raw["library_role"]
        _require(old == sequences[label][p], "site original identity disagrees with structure")
        _require(role == roles[i], "site role disagrees with source library_roles order")
        _require(isinstance(selected, str) and selected in {"H", "K", "R", "D", "E"}
                 and (role != "his" or selected == "H"), "unsupported selected identity for library role")
        role_source = _mapping(source[role], f"source.{role}")
        _require(_SOURCE_IDENTITIES.get(role_source["resname"]) == selected,
                 "selected identity differs from corresponding source atoms")
        # When detailed atom provenance is present, its chemical identity and
        # physical residue identity must agree with the named natural role.
        if "correspondence" in source:
            correspondence = _mapping(source["correspondence"], "source.correspondence")
            atom_source = _mapping(correspondence["Y10" if role == "his" else "X10"],
                                   "source atom correspondence")
            for key in ("original", "database_source"):
                record = _mapping(atom_source[key], f"source correspondence {key}")
                _require(_SOURCE_IDENTITIES.get(record["resname"]) == selected,
                         "atom correspondence identity differs from selected library role")
            original_source = atom_source["original"]
            for key in ("model", "chain", "polymer_occurrence", "resnum", "insertion_code", "segment"):
                if key in role_source:
                    _require(original_source.get(key) == role_source[key],
                             f"source {role} and atom correspondence disagree on {key}")
        policy, sidechain = raw["sequence"], raw["sidechain_policy"]
        _require((policy, sidechain) in {("mutable", "template"), ("fixed", "fixed"), ("fixed", "repack")},
                 "invalid native sequence/sidechain policy")
        _require(spec["chain"] == author and spec["sequence"] == policy and spec["sidechain"] == sidechain,
                 "site violates requested physical chain or sequence/sidechain policy")
        if spec["position"] is not None:
            _require(_integer(spec["position"], "requested site position", minimum=1) == p + 1,
                     "site violates requested position")
        if spec["residue_type"] is not None:
            identities = spec["residue_type"]
            identities = [identities] if isinstance(identities, str) else _items(identities, "requested identities")
            _require(selected in identities, "site violates requested residue_type")
        is_binder = label == binder_label
        if not is_binder:
            _require(policy == "fixed" and selected == old, "target endpoint must retain its original identity")
        elif policy == "mutable":
            _require(p in mutable, "mutable binder endpoint is outside the effective user mask")
        else:
            _require(selected == old, "fixed binder endpoint changes identity")
        if is_binder:
            pair_positions[p] = selected
            if old != selected:
                _require(p in mutable, "binder edit expands the original mutable mask")
                pair_edits[p] = (old, selected)
        user_mask = mask if is_binder else _original_mask(protomers[label_protomers[label]], len(sequences[label]))
        site = dict(chain_label=label, chain_author=author, protomer=label_protomers[label],
                    position_0based=p, position_1based=p + 1, pdb_number=rid[0], insertion_code=rid[1],
                    original_identity=old, selected_identity=selected, library_role=role,
                    sequence_policy=policy, user_designable=user_mask[p],
                    identity_fixed_for_pair=True)
        sites.append(site)

    rebuilt = list(original)
    for p, identity in pair_positions.items():
        rebuilt[p] = identity
    _require("".join(rebuilt) == designed, "pair identities do not reconstruct telemetry.designed_sequence")
    _require(bool(pair_edits), "proposed telemetry must contain a nonempty native binder pair edit")
    _validate_edits(telemetry["edits"], pair_edits, binder_numbering, native=True)
    _validate_edits(metadata["edits"], pair_edits, binder_numbering, native=False)
    _require(_integer(telemetry["n_edits"], "telemetry.n_edits") == len(pair_edits), "n_edits mismatch")
    if "placed_sidechains" in metadata:
        placed = _items(metadata["placed_sidechains"], "placed_sidechains", size=2)
        for raw, entry in zip(raw_sites, placed):
            entry = _mapping(entry, "placed_sidechain")
            _require(all(entry.get(k) == v for k, v in raw.items()),
                     "placed-sidechain atom provenance disagrees with sites")

    chain_mapping = [dict(chain_label=c, chain_author=author_map[c], protomer=p)
                     for c, p in label_protomers.items()]
    targets = {c: dict(chain_label=c, chain_author=author_map[c], protomer=label_protomers[c],
                       sequence=seq, sequence_policy="fixed", identity_fixed_for_pair=True,
                       fixed_identities_0based=dict(enumerate(seq)))
               for c, seq in sequences.items() if c != binder_label}
    centers = [dict(s, requested_microstate="HIS-S", requested_charge_state="neutral",
                    microstate_is_design_constraint=True,
                    microstate_is_experimental_assignment=False,
                    requires_target_his_s_backend_support=s["chain_label"] != binder_label)
               for s in sites if s["selected_identity"] == "H"]
    designable = mutable - set(pair_positions)
    result = dict(
        schema="vdm_pair_roles_v1", oligo_name=oligo_name,
        source=dict(structure_path=str(path.resolve()), structure_sha256=digest,
                    chain_mapping=chain_mapping, library_roles=list(roles),
                    library_source=copy.deepcopy(dict(source)),
                    position_dictionary_indexing="0-based polymer positions; JSON keys are strings"),
        binder=dict(chain_label=binder_label, chain_author=binder_author, protomer=binder_proto,
                    original_sequence=original, designed_sequence=designed,
                    mutable_positions_0based=sorted(mutable),
                    potts_designable_positions_0based=sorted(designable),
                    potts_fixed_positions_0based=sorted(set(range(len(original))) - designable)),
        targets=targets, sites=sites, his_centers=centers,
        network_pair_roles={binder_proto: dict(sorted(pair_positions.items()))},
    )
    # Provenance must be transportable too; reject unsupported/nonnumeric JSON
    # payloads instead of silently dropping evidence or changing its meaning.
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"vdm_pair_roles: result/provenance is not JSON serializable: {exc}") from exc
    return result
