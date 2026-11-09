"""Read-only, fail-closed conversion of native vdM pair sets to role locks.

Shared single-pair validation mirrors frozen ``vdm_pair_roles`` v1, allowing
explicitly selected pairs with empty edit maps. Edited pairs additionally pass
the frozen validator (SHA256
27cb64be4cec6cf94625057f3d3f3ad7b7a3a16330a58e63cbfec55c6c747330).
No proposals, mutations, sampling, configuration access, or queue writes occur.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from astrolabe.mutators import vdm_pair_roles as _v1


def _require(condition, message):
    if not condition:
        raise ValueError(f"vdm_pair_set_roles: {message}")


def build_vdm_pair_set_roles(*, oligo_name, features, telemetry, protomers) -> dict:
    """Return the complete pair set required by the resolved native mutation budget.

    Input shape (single oligomer; positions are 0-based polymer positions)::

        features = <native shared features on the original complex>
        telemetry = {
            "status": "proposed", "binder_chain": <physical label chain>,
            "structure_chain": <author chain>,
            "designed_sequence": <full combined binder candidate>,
            "projected_sequence": <same full combined candidate>,
            "n_edits": <integer>, "edits": [<native edit record>, ...],
            "metadata": {"pair_count": <integer>, "pairs": [
                <original full single-pair geometry metadata>, ...]}}

    Each native edit has ``position``, ``residue_id``, ``from`` and ``to``;
    native segment provenance, when present, is retained in per-pair telemetry.
    Each pair metadata retains its original sequences, sites, source, masks,
    edits and geometry provenance. With explicit ``pair_count=1``, existing
    native v1 telemetry (metadata without either pair-set field) is accepted.
    The features carry the resolved ``mutation_budget``; each pair costs the
    number of mutable sites in the native two-site policy. An existing fixed
    anchor costs no mutable slot and retains its original identity.
    Reported counts and the actual list length must match that derived count.
    Explicitly selected pairs may have zero sequence edits, including when
    the entire proposed set has ``n_edits=0``, empty edits and original binder
    sequence. Their endpoints, provenance and identity locks are still fully
    validated and retained. This does not make a no-match/no-op status valid.

    Output shape::

        {"schema": "vdm_pair_set_roles_v2", "oligo_name": <name>,
         "pair_count": <integer>, "source": <shared structure provenance>,
         "pairs": [{"pair_index": <0-based index>, "sites": [<site>, <site>],
                    "provenance": <full unchanged pair metadata>}, ...],
         "his_centers": [<every selected H site, with pair_index,
                          requested_microstate="HIS-S", neutral charge,
                          explicit target-His backend-support requirement>],
         "network_pair_roles": {<binder protomer>: {<integer position>: <AA>}},
         "binder": <v1 binder fields with full combined designed_sequence and
                    fixed/designable positions recomputed over ALL endpoints>,
         "targets": {<physical label chain>: <v1 fixed target identities>}}

    All selected binder endpoints are identity locks, including unchanged H
    partners. Potts designable positions are the original effective eligible
    mask minus all binder endpoints; every other binder position is fixed.
    All target sequences remain original. HIS-S is a requested constraint,
    not an experimental assignment; cross-chain target His requires explicit
    backend support. Position dictionary keys are Python integers (JSON
    serialization renders object keys as strings). The result is JSON safe.

    Raises ``ValueError`` for no-match/no-op/failure or insufficient pairs,
    malformed/noninteger counts, stale or ambiguous structure/chain mapping,
    duplicate physical endpoints across any pairs (including shared targets),
    inconsistent roles, sequences, masks, edits/counts/projections, changed
    targets, non-JSON provenance, or any failed single-pair validation. Actual
    queued binder sequence must be original or EXACTLY the full candidate;
    partial, mixed or concurrent queue changes fail closed. Each pair is
    validated independently on isolated original-sequence protomer snapshots,
    after validating the real queues. A shared v2 validator retains every
    frozen v1 check except its nonempty-edit requirement; edited pairs also
    pass frozen v1 itself. Empty native and geometry edit maps must agree with
    unchanged endpoint identities and sequences. Inputs and queues are untouched;
    no partial result, ``None`` or independently runnable Potts fallback is
    returned. Native incomplete-input errors are normalized to ``ValueError``.
    """
    try:
        return _build(oligo_name, features, telemetry, protomers)
    except (KeyError, TypeError, AttributeError, IndexError, OSError,
            OverflowError) as exc:
        raise ValueError(
            f"vdm_pair_set_roles: invalid or incomplete native proposal: {exc}"
        ) from exc


def _snapshot(protomers, oligo_name):
    """Isolate native lazy mask caches and capture the actual queue boundary."""
    snapshots, pending = {}, {}
    for name in dict.fromkeys(oligo_name):
        proto = copy.copy(protomers[name])
        original = _v1._sequence(str(proto.sequence), f"protomer {name}")
        pending[name] = _v1._sequence(str(proto.mutated_sequence),
                                     f"pending protomer {name}")
        snapshots[name] = SimpleNamespace(
            sequence=original, mutated_sequence=original,
            mutable_mask=_v1._original_mask(proto, len(original)),
        )
    return snapshots, pending


def _build(oligo_name, features, telemetry, protomers):
    _require(isinstance(oligo_name, str) and bool(oligo_name),
             "oligo_name must be nonempty")
    features = _v1._mapping(features, "features")
    telemetry = _v1._mapping(telemetry, "telemetry")
    protomers = _v1._mapping(protomers, "protomers")
    budget = _v1._integer(features["mutation_budget"], "features.mutation_budget")
    site_specs = _v1._items(features["sites"], "features.sites", size=2)
    policies = [_v1._mapping(s, "feature site")["sequence"] for s in site_specs]
    _require(all(p in ("fixed", "mutable") for p in policies), "invalid site sequence policy")
    per_pair = sum(p == "mutable" for p in policies)
    _require(per_pair > 0, "pair requires at least one mutable site")
    pair_count, remainder = divmod(budget, per_pair)
    _require(pair_count > 0, "resolved mutation budget is insufficient for a pair")
    if "pair_count" in features:
        _require(_v1._integer(features["pair_count"], "features.pair_count", minimum=1) == pair_count,
                 "feature pair_count disagrees with resolved mutation budget")
    if "unused_mutation_budget" in features:
        _require(_v1._integer(features["unused_mutation_budget"], "features.unused_mutation_budget") == remainder,
                 "feature unused budget disagrees with resolved mutation budget")
    _require(_v1._status(telemetry["status"], "telemetry.status") == "proposed",
             "exact proposed pair set required; no-match/no-op/failure has no roles")
    _require(_v1._status(telemetry.get("engine_status", "proposed"),
                         "telemetry.engine_status") == "proposed",
             "pair-set engine status must be proposed")
    metadata = _v1._mapping(telemetry["metadata"], "telemetry.metadata")
    for key, expected in (("mutation_budget", budget), ("unused_mutation_budget", remainder)):
        if key in metadata:
            _require(_v1._integer(metadata[key], f"metadata.{key}") == expected,
                     f"metadata.{key} disagrees with resolved mutation budget")
    if "pairs" in metadata:
        reported = _v1._integer(metadata["pair_count"], "metadata.pair_count", minimum=1)
        pair_metadata = _v1._items(metadata["pairs"], "metadata.pairs", size=pair_count)
        _require(reported == pair_count, "metadata.pair_count differs from requested pair_count")
    else:
        _require(pair_count == 1, "single-pair telemetry conflicts with resolved mutation budget and site policy")
        if "pair_count" in metadata:
            _require(_v1._integer(metadata["pair_count"], "metadata.pair_count", minimum=1) == 1,
                     "single-pair telemetry reports a different pair count")
        pair_metadata = [metadata]

    original = _v1._sequence(features["sequence"], "features.sequence")
    candidate = _v1._sequence(telemetry["designed_sequence"], "designed_sequence")
    _require(len(candidate) == len(original), "combined candidate changes binder length")
    _require(telemetry["projected_sequence"] == candidate,
             "projected_sequence differs from full combined candidate")
    binder_name = features["binder_protomer"]
    _require(binder_name in oligo_name and oligo_name.count(binder_name) == 1,
             "binder protomer is absent, ambiguous or repeated")
    snapshots, pending = _snapshot(protomers, oligo_name)
    _require(snapshots[binder_name].sequence == original, "stale original binder sequence")
    for name, proto in snapshots.items():
        _require(pending[name] in (original, candidate) if name == binder_name
                 else pending[name] == proto.sequence,
                 f"pending protomer {name} contains partial, mixed or target mutations")

    # Validate every full native edit before selecting per-pair subsets. The
    # frozen validator also compares residue numbering to the actual structure.
    numbering = [_v1._residue_id(r, native=True, label="features.residue_ids")
                 for r in _v1._items(features["residue_ids"], "features.residue_ids")]
    _require(len(numbering) == len(original), "binder numbering length mismatch")
    expected_edits = {p: (old, new) for p, (old, new) in enumerate(zip(original, candidate))
                      if old != new}
    _v1._validate_edits(telemetry["edits"], expected_edits, numbering, native=True)
    _require(_v1._integer(telemetry["n_edits"], "telemetry.n_edits") == len(expected_edits),
             "full telemetry n_edits mismatch")
    native_edits = {int(e["position"]): e for e in telemetry["edits"]}

    pairs, centers, locks, occupied = [], [], {}, set()
    shared = None
    rebuilt = list(original)
    for index, raw_metadata in enumerate(pair_metadata):
        raw_metadata = _v1._mapping(raw_metadata, f"metadata.pairs[{index}]")
        # Derive only the single-pair candidate; never rewrite geometry metadata
        # to make a sequential or combined-original proposal pass v1.
        pair_sequence = list(original)
        for raw in _v1._items(raw_metadata["sites"], "pair metadata.sites", size=2):
            raw = _v1._mapping(raw, "pair site")
            if raw["chain"] == features["binder_chain"]:
                p = _v1._integer(raw["position_zero_based"], "pair site position")
                _require(p < len(original), "binder pair endpoint is out of range")
                pair_sequence[p] = raw["residue_type"]
        pair_sequence = _v1._sequence("".join(pair_sequence), "single-pair candidate")
        _require(len(pair_sequence) == len(original), "single-pair candidate changes binder length")
        pair_positions = {p for p, (a, b) in enumerate(zip(original, pair_sequence)) if a != b}
        _require(pair_positions <= set(native_edits),
                 "pair edits are absent from full native telemetry")
        pair_telemetry = dict(
            status="proposed", engine_status="proposed",
            binder_chain=telemetry["binder_chain"], structure_chain=telemetry["structure_chain"],
            designed_sequence=pair_sequence, projected_sequence=pair_sequence,
            n_edits=len(pair_positions),
            edits=[copy.deepcopy(native_edits[p]) for p in sorted(pair_positions)],
            metadata=copy.deepcopy(dict(raw_metadata)),
        )
        validated = _validate_selected_pair(oligo_name, features, pair_telemetry, snapshots)
        if pair_positions:
            frozen = _v1.build_vdm_pair_roles(
                oligo_name=oligo_name, features=features,
                telemetry=pair_telemetry, protomers=snapshots,
            )
            _require(validated == frozen, "shared validation disagrees with frozen v1")
        _require(validated is not None, "a selected pair did not yield validated roles")
        if shared is None:
            shared = validated
        else:
            for key in ("structure_path", "structure_sha256", "chain_mapping"):
                _require(validated["source"][key] == shared["source"][key],
                         f"pair {index} has inconsistent {key}")
            _require(validated["targets"] == shared["targets"],
                     "pairs disagree on original target identities")
            _require(validated["binder"]["mutable_positions_0based"] ==
                     shared["binder"]["mutable_positions_0based"],
                     "pairs disagree on original eligible binder mask")
        for site in validated["sites"]:
            endpoint = (site["chain_label"], site["position_0based"])
            _require(endpoint not in occupied,
                     f"duplicate physical endpoint {endpoint!r} across selected pairs")
            occupied.add(endpoint)
            if site["chain_label"] == features["native_chain"]:
                p = site["position_0based"]
                locks[p] = site["selected_identity"]
                rebuilt[p] = site["selected_identity"]
        pairs.append(dict(pair_index=index, sites=copy.deepcopy(validated["sites"]),
                          provenance=copy.deepcopy(dict(raw_metadata))))
        centers.extend(dict(copy.deepcopy(center), pair_index=index)
                       for center in validated["his_centers"])

    _require("".join(rebuilt) == candidate,
             "all pair endpoints do not reconstruct full combined candidate")
    binder = copy.deepcopy(shared["binder"])
    designable = set(binder["mutable_positions_0based"]) - set(locks)
    binder.update(designed_sequence=candidate,
                  potts_designable_positions_0based=sorted(designable),
                  potts_fixed_positions_0based=sorted(set(range(len(original))) - designable))
    result = dict(
        schema="vdm_pair_set_roles_v2", oligo_name=oligo_name, pair_count=pair_count,
        mutation_budget=budget, unused_mutation_budget=remainder,
        source={key: copy.deepcopy(shared["source"][key]) for key in (
            "structure_path", "structure_sha256", "chain_mapping", "position_dictionary_indexing")},
        pairs=pairs, his_centers=centers,
        network_pair_roles={binder_name: dict(sorted(locks.items()))},
        binder=binder, targets=copy.deepcopy(shared["targets"]),
    )
    try:
        # Validate the complete input provenance too, including set-level data
        # that is not duplicated into every output pair.
        json.dumps(metadata, allow_nan=False)
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"vdm_pair_set_roles: non-JSON result/provenance: {exc}") from exc
    current, current_pending = _snapshot(protomers, oligo_name)
    _require(current_pending == pending and all(
        current[name].sequence == proto.sequence and current[name].mutable_mask == proto.mutable_mask
        for name, proto in snapshots.items()),
        "original sequences, masks or queued mutations changed during conversion")
    return result


def _validate_selected_pair(oligo_name, features, telemetry, protomers):
    """Apply the complete frozen-v1 validation to an explicitly selected pair.

    This is an explicit validation mirror, shared by edited and unchanged
    pairs. The sole semantic difference from frozen v1's ``_build`` is removal
    of its nonempty ``pair_edits`` requirement. Empty edits must still pass
    both native/geometry edit reconstruction and count checks. In particular,
    no source identities, masks, original sequences, structure data, policies
    or placed-sidechain provenance are synthesized or weakened for replay.
    Keep this mirror pinned to the v1 SHA documented above; v1 stays frozen.
    """
    features = _v1._mapping(features, "features")
    telemetry = _v1._mapping(telemetry, "telemetry")
    status = _v1._status(telemetry["status"], "telemetry.status")
    engine_status = _v1._status(telemetry.get("engine_status", status), "telemetry.engine_status")
    if _v1._is_noop(status):
        _v1._require(_v1._is_noop(engine_status) or engine_status == "proposed",
                 "no-op telemetry has an invalid engine status")
        original = features.get("sequence")
        for key in ("designed_sequence", "projected_sequence"):
            if key in telemetry:
                _v1._require(telemetry[key] == original, f"no-op {key} changes the binder")
        _v1._require(not telemetry.get("edits") and telemetry.get("n_edits", 0) == 0,
                 "no-op telemetry contains edits")
        return None
    _v1._require(status == "proposed" and engine_status == "proposed", "unsupported/failure proposal status")
    _v1._require(isinstance(oligo_name, str) and bool(oligo_name), "oligo_name must be nonempty")
    protomers = _v1._mapping(protomers, "protomers")
    metadata = _v1._mapping(telemetry["metadata"], "metadata")

    path = Path(features["structure_path"])
    _v1._require(path.is_file(), f"original structure does not exist: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _v1._require(metadata["structure_sha256"] == digest, "original structure SHA256 mismatch")
    # Import only for a genuine candidate. Reuse native parsing and label-chain
    # semantics, without constructing a mutator or reading its configuration.
    from astrolabe.mutators.vdm import _structure_sequences
    from astrolabe.utils.mmcif_ids import int_id_to_str_id

    try:
        sequences, numbering, author_map = _structure_sequences(path)
    except Exception as exc:
        raise ValueError(f"vdm_pair_roles: cannot parse original structure: {exc}") from exc
    _v1._require(hashlib.sha256(path.read_bytes()).hexdigest() == digest,
             "original structure changed during conversion")
    label_protomers = {int_id_to_str_id(i + 1): p for i, p in enumerate(oligo_name)}
    _v1._require(set(sequences) == set(label_protomers), "oligo physical label chains disagree with structure")
    _v1._require(set(author_map) == set(sequences)
             and all(isinstance(a, str) and bool(a) for a in author_map.values())
             and len(set(author_map.values())) == len(author_map),
             "NEED_DECISION: author-chain map must be one-to-one")
    expected_sequences = {}
    for label, name in label_protomers.items():
        _v1._require(name in protomers, f"unknown protomer {name!r}")
        expected_sequences[label] = _v1._sequence(str(protomers[name].sequence), f"protomer {name}")
    _v1._require(sequences == expected_sequences, "complex sequences differ from original protomers")
    binder_label = features["native_chain"]
    _v1._require(binder_label in sequences, "native binder label chain is absent")
    binder_author = author_map[binder_label]
    binder_proto = label_protomers[binder_label]
    _v1._require(features["binder_chain"] == binder_author
             and features["binder_protomer"] == binder_proto
             and oligo_name.count(binder_proto) == 1,
             "binder label/author/protomer mapping is inconsistent or repeated")
    _v1._require(telemetry["binder_chain"] == binder_label
             and telemetry["structure_chain"] == binder_author
             and metadata["binder_chain"] == binder_author, "telemetry binder-chain mapping disagrees")
    original = sequences[binder_label]
    _v1._require(features["sequence"] == original and features["binder_sequence"] == original
             and metadata["original_binder_sequence"] == original, "stale original binder sequence")
    designed = _v1._sequence(telemetry["designed_sequence"], "designed_sequence")
    _v1._require(len(designed) == len(original), "designed binder length changed")
    _v1._require(telemetry["projected_sequence"] == designed,
             "projected_sequence includes implicit concurrent mutations")
    pending_binder = _v1._sequence(str(protomers[binder_proto].mutated_sequence),
                               "pending binder mutated_sequence")
    _v1._require(pending_binder in (original, designed),
             "pending binder mutations are unrelated, mixed, or only a partial pair")
    for label, name in label_protomers.items():
        if label != binder_label:
            pending_target = _v1._sequence(str(protomers[name].mutated_sequence),
                                      f"pending target {label} mutated_sequence")
            _v1._require(pending_target == sequences[label],
                     f"pending target mutations change fixed physical chain {label}")
    binder_numbering = numbering[binder_label]
    feature_numbering = [_v1._residue_id(r, native=True, label="features.residue_ids")
                         for r in _v1._items(features["residue_ids"], "features.residue_ids")]
    _v1._require(feature_numbering == binder_numbering, "features binder residue numbering is stale")

    # Read only: preserve even the original protomer's lazy mask-cache state.
    mask = _v1._original_mask(protomers[binder_proto], len(original))
    mutable = _v1._positions(features["mutable_positions"], len(original), "features.mutable_positions")
    _v1._require(all(bool(mask[p]) for p in mutable), "features expands original seqs/mutable_mask constraints")
    effective = _v1._positions(metadata["effective_mutable_positions"], len(original),
                           "metadata.effective_mutable_positions")
    _v1._require(effective == mutable, "effective mutable positions differ from features")

    targets_before = {author_map[c]: seq for c, seq in sequences.items() if c != binder_label}
    _v1._require(_v1._mapping(metadata["target_sequences_before"], "target_sequences_before") == targets_before
             and _v1._mapping(metadata["target_sequences_after"], "target_sequences_after") == targets_before,
             "all actual target chains must retain their original sequences")
    if "target_sequence_unchanged" in metadata:
        _v1._require(metadata["target_sequence_unchanged"] is True, "target sequence change reported")

    raw_sites = _v1._items(metadata["sites"], "metadata.sites", size=2)
    specs = _v1._items(features["sites"], "features.sites", size=2)
    roles = _v1._items(metadata["library_roles"], "metadata.library_roles", size=2)
    _v1._require(all(isinstance(r, str) for r in roles) and set(roles) == {"his", "partner"},
             "library roles must be exactly his and partner")
    source = _v1._mapping(metadata["source"], "metadata.source")
    author_labels = {a: c for c, a in author_map.items()}
    sites, pair_positions, pair_edits = [], {}, {}
    occupied = set()
    for i, raw in enumerate(raw_sites):
        raw = _v1._mapping(raw, f"site {i}")
        spec = _v1._mapping(specs[i], f"features site {i}")
        _v1._require(_v1._integer(raw["site_index"], "site_index") == i, "site_index disagrees with endpoint order")
        author = raw["chain"]
        _v1._require(author in author_labels, "site author chain is absent from structure")
        label = author_labels[author]
        p = _v1._integer(raw["position_zero_based"], "site position_zero_based")
        _v1._require(p < len(sequences[label]) and (label, p) not in occupied,
                 "site is outside its physical chain or duplicates the other endpoint")
        occupied.add((label, p))
        _v1._require(_v1._integer(raw["position_one_based"], "site position_one_based", minimum=1) == p + 1,
                 "site 0-based/1-based positions disagree")
        rid = _v1._residue_id(raw["residue_id"], native=False, label="site.residue_id")
        _v1._require(rid == numbering[label][p], "site PDB number/insertion code disagrees with input structure")
        old, selected, role = raw["original_residue_type"], raw["residue_type"], raw["library_role"]
        _v1._require(old == sequences[label][p], "site original identity disagrees with structure")
        _v1._require(role == roles[i], "site role disagrees with source library_roles order")
        _v1._require(isinstance(selected, str) and selected in {"H", "K", "R", "D", "E"}
                 and (role != "his" or selected == "H"), "unsupported selected identity for library role")
        role_source = _v1._mapping(source[role], f"source.{role}")
        _v1._require(_v1._SOURCE_IDENTITIES.get(role_source["resname"]) == selected,
                 "selected identity differs from corresponding source atoms")
        # When detailed atom provenance is present, its chemical identity and
        # physical residue identity must agree with the named natural role.
        if "correspondence" in source:
            correspondence = _v1._mapping(source["correspondence"], "source.correspondence")
            atom_source = _v1._mapping(correspondence["Y10" if role == "his" else "X10"],
                                   "source atom correspondence")
            for key in ("original", "database_source"):
                record = _v1._mapping(atom_source[key], f"source correspondence {key}")
                _v1._require(_v1._SOURCE_IDENTITIES.get(record["resname"]) == selected,
                         "atom correspondence identity differs from selected library role")
            original_source = atom_source["original"]
            for key in ("model", "chain", "polymer_occurrence", "resnum", "insertion_code", "segment"):
                if key in role_source:
                    _v1._require(original_source.get(key) == role_source[key],
                             f"source {role} and atom correspondence disagree on {key}")
        policy, sidechain = raw["sequence"], raw["sidechain_policy"]
        _v1._require((policy, sidechain) in {("mutable", "template"), ("fixed", "fixed"), ("fixed", "repack")},
                 "invalid native sequence/sidechain policy")
        _v1._require(spec["chain"] == author and spec["sequence"] == policy and spec["sidechain"] == sidechain,
                 "site violates requested physical chain or sequence/sidechain policy")
        if spec["position"] is not None:
            _v1._require(_v1._integer(spec["position"], "requested site position", minimum=1) == p + 1,
                     "site violates requested position")
        if spec["residue_type"] is not None:
            identities = spec["residue_type"]
            identities = [identities] if isinstance(identities, str) else _v1._items(identities, "requested identities")
            _v1._require(selected in identities, "site violates requested residue_type")
        is_binder = label == binder_label
        if not is_binder:
            _v1._require(policy == "fixed" and selected == old, "target endpoint must retain its original identity")
        elif policy == "mutable":
            _v1._require(p in mutable, "mutable binder endpoint is outside the effective user mask")
        else:
            _v1._require(selected == old, "fixed binder endpoint changes identity")
        if is_binder:
            pair_positions[p] = selected
            if old != selected:
                _v1._require(p in mutable, "binder edit expands the original mutable mask")
                pair_edits[p] = (old, selected)
        user_mask = mask if is_binder else _v1._original_mask(protomers[label_protomers[label]], len(sequences[label]))
        site = dict(chain_label=label, chain_author=author, protomer=label_protomers[label],
                    position_0based=p, position_1based=p + 1, pdb_number=rid[0], insertion_code=rid[1],
                    original_identity=old, selected_identity=selected, library_role=role,
                    sequence_policy=policy, user_designable=user_mask[p],
                    identity_fixed_for_pair=True)
        sites.append(site)

    rebuilt = list(original)
    for p, identity in pair_positions.items():
        rebuilt[p] = identity
    _v1._require("".join(rebuilt) == designed, "pair identities do not reconstruct telemetry.designed_sequence")
    _v1._validate_edits(telemetry["edits"], pair_edits, binder_numbering, native=True)
    _v1._validate_edits(metadata["edits"], pair_edits, binder_numbering, native=False)
    _v1._require(_v1._integer(telemetry["n_edits"], "telemetry.n_edits") == len(pair_edits), "n_edits mismatch")
    if "placed_sidechains" in metadata:
        placed = _v1._items(metadata["placed_sidechains"], "placed_sidechains", size=2)
        for raw, entry in zip(raw_sites, placed):
            entry = _v1._mapping(entry, "placed_sidechain")
            _v1._require(all(entry.get(k) == v for k, v in raw.items()),
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
