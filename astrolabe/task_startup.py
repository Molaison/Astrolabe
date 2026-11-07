"""Import-cheap common task checks, before any scientific runtime is loaded.

Only selected input leaves are read. In particular, resolving the whole task
would also resolve inactive model defaults. This module uses only the standard
library and never parses structures, samples sequences, or changes the config.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import os
from pathlib import Path
import re


_TYPE_ALIASES = {
    "p": "p", "prot": "p", "protein": "p", "r": "r", "rna": "r",
    "d": "d", "dna": "d", "s": "s", "smiles": "s", "c": "c", "ccd": "c",
}
_BOOLEAN_FIELDS = (
    "debug", "enable_pyrosetta", "skip_complex_poses", "allow_mmcif_pdb_fallback",
    "use_metropolis_acceptance", "dynamic_binding_site_enabled",
)
_EXPRESSIONS = (
    "weighted_score_expression", "acceptance_gate", "acceptance_threshold",
    "early_stopping", "prefix_early_stopping", "_entrance",
)
_MISSING = object()
_SEQUENCE_SEARCHES = frozenset({
    ("branching", "isolated_native"),
    ("branching", "native_refinement"),
    ("branching", "archived_refinement"),
})
_EXECUTION_FIELDS = frozenset({
    "socket_path", "workers", "in_flight", "resume", "retry_interrupted",
})


def validate_task_config(cfg, *, raw, resolve_environment=True, require_inputs=False,
                         error_type=ValueError, reference_info=None):
    """Reject explicit common-field errors; omitted API-only fields stay omitted.

    ``raw`` is the caller's unresolved plain mapping. ``cfg`` retains the root
    interpolation context. Dry-run leaves containing interpolations are deferred;
    independent literal siblings are still checked. Error construction is supplied
    by the caller so this module has no dependency on the startup coordinator.

    Missing entry-point inputs are returned for coordinator aggregation, rather
    than masking existing diagnostics for otherwise valid partial API configs.
    """
    errors = []
    required_errors = []

    def fail(path, message):
        errors.append(f"{path}: {message}")

    def deferred(value):
        return not resolve_environment and isinstance(value, str) and "${" in value

    def read(*keys):
        node = cfg if resolve_environment else raw
        try:
            for key in keys:
                if isinstance(node, Mapping):
                    # DictConfig membership excludes mandatory-missing values;
                    # its keys still include them, so the access reports ???.
                    if key not in node.keys():
                        return _MISSING
                elif not (isinstance(node, Sequence) and isinstance(key, int)):
                    return _MISSING
                node = node[key]
        except Exception as exc:
            fail(".".join(map(str, keys)), str(exc))
            return _MISSING
        if deferred(node):
            return _MISSING
        if isinstance(node, str) and node == "???":
            fail(".".join(map(str, keys)), "mandatory value is missing")
            return _MISSING
        return node

    def mapping(*keys, nullable=False):
        value = read(*keys)
        if value is _MISSING or value is None and nullable:
            return {}
        if not isinstance(value, Mapping):
            fail(".".join(map(str, keys)), "expected a mapping")
            return {}
        return value

    def is_list(value):
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))

    def selected(*keys):
        value = read(*keys)
        if isinstance(value, Mapping):
            return {key: selected(*keys, key) for key in value.keys()}
        if is_list(value):
            return [selected(*keys, index) for index in range(len(value))]
        return value

    def boolean(value, path):
        if value is not _MISSING and type(value) is not bool:
            fail(path, "expected a boolean")

    def number(value, path, *, minimum=None, integer=False):
        if value is _MISSING:
            return None
        try:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError()
            result = float(value)
            if not math.isfinite(result) or minimum is not None and result < minimum:
                raise ValueError()
            if integer and (int(value) != result):
                raise ValueError()
            return result
        except (ValueError, TypeError, OverflowError):
            description = "integer" if integer else "number"
            suffix = f" >= {minimum}" if minimum is not None else ""
            fail(path, f"expected a finite {description}{suffix}")
            return None

    def path_value(value, path):
        if value is _MISSING:
            return
        if not isinstance(value, (str, os.PathLike)) or not os.fspath(value).strip():
            fail(path, "expected a non-empty file path")
            return
        # Like backend preflight, composition-only dry runs do not require assets.
        if resolve_environment and not raw.get("dry_run"):
            try:
                if not Path(value).is_file():
                    fail(path, f"file does not exist or is not a regular file: {value}")
            except (OSError, ValueError) as exc:
                fail(path, f"invalid file path: {exc}")

    for key in _BOOLEAN_FIELDS:
        boolean(read(key), key)
    name = read("task_name")
    if name is not _MISSING and (name is not None or require_inputs):
        if not isinstance(name, str) or not name.strip():
            fail("task_name", "expected a non-empty task name string")
    for key in ("out_dir", "raw_model_results_dir"):
        value = read(key)
        if value is _MISSING or value is None or value == "":
            continue
        if not isinstance(value, (str, os.PathLike)):
            fail(key, "expected a directory path or null")
        elif resolve_environment and not raw.get("dry_run"):
            try:
                target = Path(value).expanduser()
                # No writes or probes: a known file in the directory ancestry
                # cannot become a run directory.
                for parent in (target, *target.parents):
                    if parent.exists():
                        if not parent.is_dir():
                            fail(key, f"directory path is blocked by a file: {parent}")
                        break
            except (OSError, ValueError) as exc:
                fail(key, f"invalid directory path: {exc}")
    excluded = read("exclude_AA")
    if excluded is not _MISSING and excluded is not None:
        alphabet = set("ACDEFGHIKLMNPQRSTVWY")
        if not is_list(excluded) or any(not isinstance(x, str) or x not in alphabet for x in excluded):
            fail("exclude_AA", "expected a list of single standard amino-acid letters")
        elif set(excluded) == alphabet:
            fail("exclude_AA", "must leave at least one amino acid available for sampling")
    # calculate_temperature explicitly maps non-positive temperatures to zero.
    if read("use_metropolis_acceptance") is True:
        temperature = read("initial_temperature")
        if temperature is _MISSING and "initial_temperature" not in raw:
            fail("initial_temperature", "required when use_metropolis_acceptance is true")
        else:
            number(temperature, "initial_temperature")

    sequence_driven = False
    if require_inputs:
        search = read("search")
        sequence_driven = "search" not in raw or search is None
        if isinstance(search, Mapping):
            algorithm, owner = read("search", "algorithm"), read("search", "owner")
            if isinstance(algorithm, str) and isinstance(owner, str):
                sequence_driven = (algorithm.strip(), owner.strip()) in _SEQUENCE_SEARCHES
        if sequence_driven and "seqs" not in raw:
            required_errors.append("seqs: required for a built-in sequence-driven workflow")
    seqs = mapping("seqs")
    if "seqs" in raw and read("seqs") is not _MISSING and not seqs:
        fail("seqs", "expected a non-empty protomer-to-sequence mapping")
    types = mapping("types", nullable=True)
    files = mapping("named_files", nullable=True)
    homologies = mapping("homologies")
    for alias in files:
        path_value(read("named_files", alias), f"named_files.{alias}")
    for proto in types:
        kind = read("types", proto)
        if kind is not _MISSING and (not isinstance(kind, str) or kind.lower() not in _TYPE_ALIASES):
            fail(f"types.{proto}", "expected a supported protomer type (protein/RNA/DNA/SMILES/CCD)")

    # Mirror only the basic literal/range/alias dispatch in Protomer.initialize.
    # PTM chemistry, file contents, CCD and SMILES parsing remain with consumers.
    aliases = {}
    for proto in seqs:
        if not isinstance(proto, str) or re.fullmatch(r"[A-Za-z+]", proto) is None:
            fail(f"seqs.{proto}", "expected a single A-Z/a-z/+ protomer identifier")
        statement = read("seqs", proto)
        kind = read("types", proto)
        if kind is _MISSING:
            # A deferred type must not accidentally be treated as protein.
            if proto in types:
                continue
            kind = "p"
        if not isinstance(kind, str) or kind.lower() not in _TYPE_ALIASES:
            continue
        kind = _TYPE_ALIASES[kind.lower()]
        if statement is _MISSING:
            continue
        if not isinstance(statement, str) or not statement.strip():
            fail(f"seqs.{proto}", "expected a non-empty sequence statement string")
            continue
        for index, token in enumerate(statement.split(), 1):
            if kind in {"s", "c"}:
                # The consumer deliberately accepts arbitrary chemical literals.
                continue
            segment, *names = token.split("@")
            segment = segment.removeprefix("+")
            clean = segment.replace("+", "").split(":", 1)[0]
            if clean in aliases and aliases[clean] == kind:
                pass
            elif segment and set(segment) <= set("0123456789-+"):
                match = re.fullmatch(r"(\d+)-(\d+)", segment)
                if match is None or int(match[2]) < int(match[1]):
                    fail(f"seqs.{proto}", "invalid random length range; expected nonnegative min-max with min <= max")
            elif clean in files:
                if kind != "p":
                    fail(f"seqs.{proto}", "structure-backed segments support only protein protomers")
                elif ":" not in segment or not segment.split(":", 1)[1]:
                    fail(f"seqs.{proto}", "file segment requires alias:chain or alias:chain-range")
            elif ":" in segment:
                fail(f"seqs.{proto}", f"undefined named-file alias {clean!r}")
            else:
                literal = re.sub(r"[\[\(].+?[\]\)]", "?", clean)
                alphabet = {"p": "ACDEFGHIKLMNPQRSTVWY?X", "r": "ACGU?", "d": "ACGT?"}[kind]
                if not literal or any(base not in alphabet for base in literal):
                    fail(f"seqs.{proto}", "invalid sequence segment; expected literal, min-max, named-file or earlier segment alias")
            aliases[f"{proto}_{index}"] = kind
            aliases.update({name: kind for name in names if name})

    def templates(value, path):
        if not is_list(value):
            fail(path, "expected a template list")
            return
        for index in range(len(value)):
            # Values here are already resolved, or the unresolved raw containers.
            entry = value[index]
            entry_path = f"{path}.{index}"
            if entry is _MISSING or deferred(entry):
                continue
            if not isinstance(entry, Mapping):
                fail(entry_path, "expected a template mapping")
                continue
            for key in ("queryIndices", "templateIndices"):
                indices = entry.get(key, _MISSING)
                if indices is _MISSING and key in entry or deferred(indices):
                    continue
                if not is_list(indices) or any(type(i) is not int or i < 0 for i in indices):
                    fail(f"{entry_path}.{key}", "expected a list of nonnegative integer indices")
            q, t = entry.get("queryIndices"), entry.get("templateIndices")
            if is_list(q) and is_list(t) and len(q) != len(t):
                fail(entry_path, "queryIndices and templateIndices must have equal length")
            mmcif = entry.get("mmcif")
            if mmcif is not None and mmcif is not _MISSING and not deferred(mmcif) and not isinstance(mmcif, str):
                fail(f"{entry_path}.mmcif", "expected mmCIF text or null")
            if "includeMutableResidues" in entry and entry["includeMutableResidues"] is not _MISSING and not deferred(entry["includeMutableResidues"]):
                boolean(entry["includeMutableResidues"], f"{entry_path}.includeMutableResidues")

    def homology(value, path, key=None):
        if value is _MISSING or value is None or deferred(value):
            return
        if isinstance(value, str):
            if key in {"unpairedMsa", "pairedMsa"} and value.lstrip().startswith(">"):
                return
            filename, separator, chain = value.partition(":")
            if filename.endswith(".json"):
                if not separator or not chain.strip():
                    fail(path, "AF3 JSON homology requires path.json:chain")
            elif not filename.endswith(".a3m"):
                fail(path, "expected .a3m path or AF3 .json:chain")
                return
            path_value(filename, path)
        elif is_list(value):
            if key not in {None, "templates"}:
                fail(path, "MSA input must be a path, inline FASTA string, mapping or null")
            else:
                templates(value, path)
        elif isinstance(value, Mapping):
            # Specialized antibody framework inputs are owned by their parser.
            if value.get("_type") == "antibody_framework_homology":
                return
            if "maskMutableResidues" in value and value["maskMutableResidues"] is not _MISSING and not deferred(value["maskMutableResidues"]):
                boolean(value["maskMutableResidues"], f"{path}.maskMutableResidues")
            for child in ("unpairedMsa", "pairedMsa", "templates"):
                if child in value:
                    homology(value[child], f"{path}.{child}", child)
        else:
            fail(path, "expected homology path, mapping, template list or null")

    for proto in seqs.keys() & homologies.keys():
        homology(selected("homologies", proto), f"homologies.{proto}")

    stages = mapping("stages", nullable=True)
    default = selected("stages", "default")
    if sequence_driven and not isinstance(default, Mapping):
        required_errors.append("stages.default: required for a built-in sequence-driven workflow")
    default = default if isinstance(default, Mapping) else {}
    for name in stages:
        stage = selected("stages", name)
        if stage is _MISSING or stage is None:
            continue
        if not isinstance(stage, Mapping):
            fail(f"stages.{name}", "expected a mapping")
            continue
        prefix = f"stages.{name}"
        for field in _EXPRESSIONS:
            value = stage.get(field, _MISSING)
            if value is not _MISSING and value is not None and not isinstance(value, (str, int, float)):
                fail(f"{prefix}.{field}", "expected an expression string or scalar")
            elif isinstance(value, float) and not math.isfinite(value):
                fail(f"{prefix}.{field}", "expected a finite expression scalar")
        for field in ("oligo", "ignored_complexes", "negative_design_complexes", "mutators", "active_loss_terms"):
            value = stage.get(field, _MISSING)
            if value is _MISSING or value is None:
                continue
            if not is_list(value) or not all(isinstance(item, str) and item.strip() for item in value):
                fail(f"{prefix}.{field}", "expected a list of non-empty strings")
                continue
            if field == "oligo":
                seen = set()
                for item in value:
                    if deferred(item):
                        continue
                    if re.fullmatch(r"[-~]?[A-Za-z+]+", item) is None:
                        fail(f"{prefix}.oligo", "expected optional '-'/'~' prefix followed by A-Z/a-z/+")
                        continue
                    body = item[1:] if item.startswith(("-", "~")) else item
                    normalized = "".join(sorted(body))
                    if normalized in seen:
                        fail(f"{prefix}.oligo", "duplicate oligomer composition (including prefixed/permuted forms)")
                    seen.add(normalized)
                    if seqs and set(body) - seqs.keys():
                        fail(f"{prefix}.oligo", f"undefined protomers {sorted(set(body) - seqs.keys())}")
        mutation_num = read("stages", name, "mutation_num")
        if isinstance(mutation_num, bool):
            fail(f"{prefix}.mutation_num", "expected numeric count or percentage, not boolean")
        evaluator = read("stages", name, "evaluator")
        if evaluator is not _MISSING and evaluator is not None and (not isinstance(evaluator, str) or not evaluator.strip()):
            fail(f"{prefix}.evaluator", "expected a non-empty evaluator alias string")
        weights = stage.get("mutator_weights", _MISSING)
        if weights is _MISSING or weights is None:
            continue
        if not is_list(weights):
            fail(f"{prefix}.mutator_weights", "expected a list of numeric weights")
            continue
        mutators = stage.get("mutators")
        if mutators is None:
            mutators = default.get("mutators")
        if is_list(mutators) and len(weights) != len(mutators):
            fail(f"{prefix}.mutator_weights", "length must match the effective stage mutators")
        numeric = [number(w, f"{prefix}.mutator_weights.{i}", minimum=0) if w is not _MISSING and not deferred(w) else None for i, w in enumerate(weights)]
        if all(w is not None for w in numeric):
            total = sum(numeric)
            if not math.isfinite(total) or total <= 0:
                fail(f"{prefix}.mutator_weights", "sum must be positive and finite")

    selected_mutators = set()
    for name in stages:
        stage = selected("stages", name)
        if isinstance(stage, Mapping) and is_list(stage.get("mutators")):
            selected_mutators.update(a for a in stage["mutators"] if isinstance(a, str))
    mutator = mapping("mutator", nullable=True)
    selector = read("mutator", "selector_expression")
    if selector is not _MISSING and selector is not None and not isinstance(selector, str):
        fail("mutator.selector_expression", "expected a selector expression string or null")
    terms = mutator.get("mutator_terms")
    if isinstance(terms, Mapping):
        for alias in selected_mutators & terms.keys():
            term = read("mutator", "mutator_terms", alias)
            if isinstance(term, Mapping):
                selector = read("mutator", "mutator_terms", alias, "position_selector")
                if selector is not _MISSING and selector is not None and not isinstance(selector, str):
                    fail(f"mutator.mutator_terms.{alias}.position_selector", "expected a selector expression string or null")
    expressions = mapping("loss", "expression_terms", nullable=True)
    for alias in expressions:
        value = read("loss", "expression_terms", alias)
        if value is not _MISSING and value is not None and not isinstance(value, (str, int, float)):
            fail(f"loss.expression_terms.{alias}", "expected an expression string or scalar")
    initializer = mapping("initializer", nullable=True)
    mode = read("initializer", "mode")
    if mode is not _MISSING and mode is not None and not isinstance(mode, str):
        fail("initializer.mode", "expected an initializer name string")
    if isinstance(mode, str) and mode.strip().lower() == "dplm":
        alias = read("initializer", "dplm_alias")
        if alias is not _MISSING and alias is not None and not isinstance(alias, str):
            fail("initializer.dplm_alias", "expected a mutator alias string or null")

    search = mapping("search", nullable=True)
    if search:
        for key in ("algorithm", "owner"):
            value = read("search", key)
            if value is _MISSING and key not in search:
                fail(f"search.{key}", "required for explicit search selection")
            elif value is not _MISSING and (not isinstance(value, str) or not value.strip()):
                fail(f"search.{key}", "expected a non-empty selection string")
    # Summary and resume are consumed by generic routing/repeat planning.
    summary = read("search", "summary_format")
    if summary is not _MISSING and summary is not None and str(summary).strip().lower() != "json":
        fail("search.summary_format", "expected null or 'json'")
    boolean(read("search", "resume"), "search.resume")
    native = read("search", "algorithm") == "branching" and read("search", "owner") == "native_refinement"
    if native:
        for key in ("width", "proposals_per_parent", "max_evaluations", "proposal_cap"):
            value = read("search", key)
            if value is None and key == "proposal_cap":
                continue
            if value is not _MISSING and (type(value) is not int or value < 1):
                fail(f"search.{key}", "expected a positive integer")
        if read("search", "resume") is True:
            output = read("out_dir")
            if not isinstance(output, str) or not output.strip():
                fail("search.resume", "requires out_dir to be the original run root")
            elif resolve_environment and not raw.get("dry_run"):
                root = Path(output).expanduser()
                required = (".framework/journal.sqlite", ".framework/assets.json",
                            "config.yaml", "provenance.json", "01_evaluations.jsonl")
                missing = [name for name in required if not (root / name).is_file()]
                if missing:
                    fail("search.resume", f"missing durable run files under {root}: {', '.join(missing)}")
            if read("repeats") not in (_MISSING, 1):
                fail("search.resume", "requires exactly one repeat")
    isolated = read("search", "algorithm") == "branching" and read("search", "owner") == "isolated_native"
    if isolated:
        execution = read("search", "execution")
        if execution is _MISSING:
            if "execution" not in search.keys():
                fail("search.execution", "required execution mapping for isolated_native")
        elif not isinstance(execution, Mapping):
            fail("search.execution", "expected an execution mapping for isolated_native")
        else:
            configured_fields = set(execution.keys())
            for key in sorted(_EXECUTION_FIELDS - configured_fields):
                fail(f"search.execution.{key}", "required execution field for isolated_native")
            for key in sorted(configured_fields - _EXECUTION_FIELDS, key=str):
                fail(f"search.execution.{key}", "unknown execution field for isolated_native")
        counts = {}
        for key in ("workers", "in_flight"):
            value = read("search", "execution", key)
            if value is not _MISSING:
                if type(value) is not int or value < 1:
                    fail(f"search.execution.{key}", "expected a positive integer")
                else:
                    counts[key] = value
        if counts.get("in_flight", 1) > counts.get("workers", math.inf):
            fail("search.execution.in_flight", "must be <= search.execution.workers")
        for key in ("resume", "retry_interrupted"):
            boolean(read("search", "execution", key), f"search.execution.{key}")
        if read("search", "execution", "retry_interrupted") is True and read("search", "execution", "resume") is not True:
            fail("search.execution.retry_interrupted", "requires search.execution.resume=true")
        socket = read("search", "execution", "socket_path")
        if require_inputs and socket is None:
            fail("search.execution.socket_path", "required non-empty socket path for an actual isolated_native run")
        elif socket is not _MISSING and socket is not None and (not isinstance(socket, str) or not socket.strip()):
            fail("search.execution.socket_path", "expected a non-empty socket path or null before service binding")
        settings = mapping("protein_beam", nullable=True)
        for key in ("width", "proposals_per_parent", "max_evaluations", "max_generations", "max_total_mutations", "proposal_batch_size", "workers", "in_flight"):
            value = read("protein_beam", key)
            if value is not None or key in {"width", "proposals_per_parent", "max_evaluations", "workers", "in_flight"}:
                number(value, f"protein_beam.{key}", minimum=1, integer=True)
        boolean(read("protein_beam", "pipeline_proposals"), "protein_beam.pipeline_proposals")
        for key, supported in (("selection", {"weighted", "pareto"}), ("holes_ranking", {"raw_sum", "per_atom"})):
            value = read("protein_beam", key)
            if value is not _MISSING and (not isinstance(value, str) or value not in supported):
                fail(f"protein_beam.{key}", f"expected one of {sorted(supported)}")
        gates = mapping("protein_beam", "gates", nullable=True)
        known_groups = {"geometry_max", "root_metric_max_ratio", "root_metric_min_ratio", "root_metric_max_delta"}
        metrics = {"interface_ipae", "interface_ipsae", "interface_sc", "interface_holes", "interface_buns"}
        geometry = {"target_ca_rmsd", "binder_pose_ca_rmsd", "binder_self_ca_rmsd", "interface_bb_rmsd", "root_binder_pose_ca_rmsd", "root_interface_bb_rmsd", "root_complex_bb_rmsd", "root_interface_binder_bb_rmsd"}
        for group in gates:
            if group not in known_groups:
                fail(f"protein_beam.gates.{group}", "unknown beam gate group")
                continue
            limits = mapping("protein_beam", "gates", group)
            for key in limits:
                path = f"protein_beam.gates.{group}.{key}"
                if key not in (geometry if group == "geometry_max" else metrics):
                    fail(path, "unknown geometry/metric gate")
                result = number(read("protein_beam", "gates", group, key), path, minimum=0)
                if result == 0:
                    fail(path, "expected a positive finite exploration limit")
        # Budget paths are outputs, not assets: never create or stat their files.
        policy = mapping("protein_beam", "af3_call_budget", nullable=True)
        if policy:
            for key in ("max_total", "max_per_case", "ledger", "scope"):
                if key not in policy:
                    fail(f"protein_beam.af3_call_budget.{key}", "required by the AF3 call budget")
            for key in ("max_total", "max_per_case"):
                number(read("protein_beam", "af3_call_budget", key), f"protein_beam.af3_call_budget.{key}", minimum=1, integer=True)
            for key in ("ledger", "scope"):
                value = read("protein_beam", "af3_call_budget", key)
                if value is not _MISSING and (not isinstance(value, str) or not value.strip()):
                    fail(f"protein_beam.af3_call_budget.{key}", "expected a non-empty string")

    if not errors:
        from astrolabe.task_references import validate_references
        reference_values = {
            key: selected(key) if resolve_environment else raw[key] for key in (
                "seqs", "types", "named_files", "stages", "binding_site",
                "structure_start", "ligand_resnames",
            ) if key in raw
        }
        inventory = validate_references(
            reference_values, check_paths=bool(resolve_environment and not raw.get("dry_run")),
            fail=fail)
        if reference_info is not None and inventory is not None:
            reference_info["segments"] = inventory[1]
    if errors:
        raise error_type("Startup configuration errors:\n- " + "\n- ".join(errors + required_errors))
    return required_errors
