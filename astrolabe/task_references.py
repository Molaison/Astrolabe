"""Task input and cross-reference checks without model/runtime construction."""
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import re

from astrolabe.core.binding_site import parse_binding_site
from astrolabe.core.chain_resolution import chain_to_protomer, resolve_selector_chain_ids


def _unresolved(value):
    return type(value) is object or isinstance(value, str) and ("${" in value or value == "???")


def _contains_unresolved(value):
    if isinstance(value, Mapping):
        return any(_contains_unresolved(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_unresolved(v) for v in value)
    return _unresolved(value)


def sequence_inventory(seqs, types, files, *, check_paths, fail):
    """Return known length upper bounds and segment aliases; never sample RNG.

    Unknown lengths stay unknown. Structure sequences use the exact lightweight
    parser used by Protomer. Atom geometry and chemistry remain runtime checks.
    """
    from astrolabe.core.sequence_ranges import parse_ranges
    from astrolabe.core.structure_parser import create_structure_parser

    bounds, segments, structures = {}, {}, {}

    def structure(alias, path):
        filename = files.get(alias)
        if not check_paths or not isinstance(filename, str) or _unresolved(filename):
            return None
        if alias not in structures:
            try:
                structures[alias] = create_structure_parser(filename).parse()
                if not structures[alias]:
                    fail(f"named_files.{alias}", "structure contains no readable residues")
            except (OSError, ValueError, TypeError, UnicodeError) as exc:
                fail(f"named_files.{alias}", f"cannot read structure: {exc}")
                structures[alias] = {}
        return structures[alias]

    for proto, statement in seqs.items():
        if not isinstance(statement, str) or _unresolved(statement):
            continue
        kind = str(types.get(proto, "p")).lower()
        if _unresolved(types.get(proto, "p")):
            continue
        if kind in {"s", "smiles", "c", "ccd"}:
            continue
        total = 0
        for i, token in enumerate(statement.split(), 1):
            token, *aliases = token.split("@")
            token = token.removeprefix("+")
            clean = token.replace("+", "").split(":", 1)[0]
            length = None
            if clean in segments:
                length = segments[clean]
            elif re.fullmatch(r"\d+-\d+", token):
                length = int(token.split("-")[1])
            elif clean in files and ":" in token:
                path = f"seqs.{proto}"
                try:
                    ranges = parse_ranges(token.split(":", 1)[1])
                    data = structure(clean, path)
                    if data:
                        length = 0
                        for chain, indices in ranges.items():
                            if chain not in data:
                                fail(path, f"structure {clean!r} has no chain {chain!r}")
                                continue
                            available, sequence = data[chain]
                            missing = set(indices) - set(available)
                            if missing:
                                fail(path, f"residues {sorted(missing)} not found in structure {clean!r} chain {chain!r}")
                            length += len(indices) if indices else len(sequence)
                except (ValueError, TypeError) as exc:
                    fail(path, str(exc))
            elif ":" not in token:
                literal = re.sub(r"[\[\(].+?[\]\)]", "?", clean)
                if literal and re.fullmatch("[A-Z?]+", literal):
                    length = len(literal)
            segments[f"{proto}_{i}"] = length
            segments.update({alias: length for alias in aliases if alias})
            total = total + length if total is not None and length is not None else None
        bounds[proto] = total
    return bounds, segments, structure


def validate_references(values, *, check_paths, fail):
    """Only validate fields supplied by task_startup, not inactive model defaults."""
    seqs = values.get("seqs") or {}
    types = values.get("types") or {}
    files = values.get("named_files") or {}
    if not all(isinstance(v, Mapping) for v in (seqs, types, files)):
        return
    bounds, segments, structure = sequence_inventory(
        seqs, types, files, check_paths=check_paths, fail=fail)
    stages = values.get("stages") or {}
    oligos = set()
    if isinstance(stages, Mapping):
        for stage in stages.values():
            if isinstance(stage, Mapping):
                entries = stage.get("oligo")
                if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
                    continue
                for oligo in entries:
                    if isinstance(oligo, str) and re.fullmatch(r"[-~]?[A-Za-z+]+", oligo):
                        oligos.add(oligo.lstrip("-~"))

    def site_check(raw_site, path, targets):
        if not isinstance(raw_site, Mapping):
            fail(path, "expected a binder/target mapping")
            return
        if _contains_unresolved(raw_site):
            return
        try:
            spec = parse_binding_site(raw_site)
        except (ValueError, TypeError, IndexError, AttributeError) as exc:
            fail(path, str(exc))
            return
        if spec is None:
            return
        for field, selected in (("binder", spec.binder), ("target", spec.target)):
            for oligo in targets:
                chains = resolve_selector_chain_ids(oligo, selected.chain)
                if not chains:
                    fail(f"{path}.{field}", f"selector {selected.chain!r} matches no chain/protomer in {oligo!r}")
                for chain in chains:
                    proto = chain_to_protomer(oligo)[chain]
                    maximum = bounds.get(proto)
                    if maximum is not None and any(p > maximum for p in selected.positions):
                        fail(f"{path}.{field}", f"residue position exceeds sequence length upper bound {maximum} for {oligo}:{chain}")

    site = values.get("binding_site")
    if site is not None and not _unresolved(site):
        if not isinstance(site, Mapping):
            fail("binding_site", "expected a mapping")
        elif "binder" in site or "target" in site:
            site_check(site, "binding_site", oligos)
        else:
            # Partial maps are valid: a complex can deliberately have no site.
            for oligo, config in site.items():
                if oligos and oligo not in oligos:
                    continue  # Unselected per-complex settings are not consumed.
                elif config is not None:
                    site_check(config, f"binding_site.{oligo}", [oligo])

    raw_start = values.get("structure_start")
    if isinstance(raw_start, Mapping) and not _contains_unresolved(raw_start):
        from astrolabe.structure_start_config import StructureStartConfig
        try:
            start = StructureStartConfig.from_raw(raw_start)
        except (ValueError, TypeError):
            start = None  # Existing coordinator reports schema errors precisely.
        if start:
            for oligo, ref in start.references.items():
                path = f"structure_start.references.{oligo}"
                if set(oligo) - seqs.keys():
                    fail(path, "references unknown protomers")
                if ref.structure_alias not in files:
                    fail(path + ".structure_alias", f"undefined named-file alias {ref.structure_alias!r}")
                    continue
                if sorted(ref.chain_map.values()) != sorted(oligo):
                    fail(path + ".chain_map", "must map input chains to exactly the oligomer protomers")
                data = structure(ref.structure_alias, path)
                if data is not None:
                    for chain, proto in ref.chain_map.items():
                        if chain not in data:
                            fail(path + ".chain_map", f"reference has no chain {chain!r}")
                        elif bounds.get(proto) is not None and len(data[chain][1]) > bounds[proto]:
                            fail(path, f"reference chain {chain!r} exceeds protomer {proto!r} length")

    ligand = values.get("ligand_resnames")
    if ligand is not None and not _unresolved(ligand):
        if not isinstance(ligand, Mapping):
            fail("ligand_resnames", "expected a residue-name mapping")
        elif any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
                 for k, v in ligand.items()):
            fail("ligand_resnames", "expected non-empty string keys and residue names")
    return oligos, segments


def validate_loss_references(options, *, path, oligos, segments, fail):
    target = options.get("_for", options.get("oligomer_specific"))
    targets = target.split(",") if isinstance(target, str) else target
    if targets is not None:
        if not isinstance(targets, (list, tuple)) or not all(isinstance(x, str) for x in targets):
            fail(path + "._for", "expected an oligomer string or list")
        elif oligos and set(targets) - oligos:
            fail(path + "._for", f"unknown stage oligomers {sorted(set(targets) - oligos)}")
    # These consumers use segment aliases, not complex-chain selectors.
    if options.get("_type") not in {"hotspot_dsasa", "hotspot_dsasa_fraction", "hotspot_contact_support"}:
        return
    expression = options.get("select_expression")
    if not isinstance(expression, str) or _unresolved(expression):
        return
    for piece in expression.split(";"):
        alias, _, indices = piece.strip().partition(":")
        if not alias:
            continue
        if alias not in segments:
            fail(path + ".select_expression", f"unknown segment alias {alias!r}")
            continue
        indices, _, weight = indices.partition("@")
        if weight:
            try:
                if not math.isfinite(float(weight)):
                    raise ValueError()
            except ValueError:
                fail(path + ".select_expression", "expected a finite selector weight")
        if indices in {"", "all"}:
            continue
        for item in indices.split(","):
            match = re.fullmatch(r"(\d+)(?:-(\d+))?", item.strip())
            if not match:
                fail(path + ".select_expression", f"invalid residue range {item!r}")
                continue
            lo, hi = int(match[1]), int(match[2] or match[1])
            if lo < 1 or hi < lo or segments[alias] is not None and hi > segments[alias]:
                fail(path + ".select_expression", f"residue range {item!r} outside segment {alias!r}")
