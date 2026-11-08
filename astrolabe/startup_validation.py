"""Configuration checks that must run before importing scientific runtimes."""
from __future__ import annotations

import ast
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
import math
import os
import sys
from copy import deepcopy

from omegaconf import OmegaConf
from astrolabe.backend_startup import validate_backend_settings
from astrolabe.scoring_startup import validate_scoring_settings
from astrolabe.component_config import merge_component_config
from astrolabe.loss_config import INTRINSIC_PARAMS
from astrolabe.stage_config import resolve_mutation_num
from astrolabe.task_startup import validate_task_config


_STAGE_EXPRESSIONS = frozenset({
    "weighted_score_expression", "acceptance_gate", "acceptance_threshold",
    "early_stopping", "prefix_early_stopping", "_entrance",
})
_STAGE_FIELDS = frozenset({
    "oligo", "weighted_score_expression", "acceptance_gate", "acceptance_threshold",
    "early_stopping", "prefix_early_stopping", "ignored_complexes",
    "negative_design_complexes", "evaluator", "active_loss_terms", "mutators",
    "mutator_weights", "transition_consistency_gate", "mutation_num", "_entrance",
})
_MUTATOR_DEFAULTS = {
    "mcmc": ("MCMC",), "pore_blocker_mcmc": ("MCMC",),
    "ligandmpnn": ("LigandMPNN",), "grouped_ligandmpnn": ("LigandMPNN",),
    "proteinmpnn": ("LigandMPNN",), "multistateproteinmpnn": ("LigandMPNN",),
    "multistate_proteinmpnn": ("LigandMPNN",),
    "faraparttiedmultitargetligandmpnn": ("LigandMPNN",),
    "far_apart_tied_multitarget_ligandmpnn": ("LigandMPNN",),
    "dynamicmpnn": ("DynamicMPNN",), "surfdesign": ("SurfDesign",),
    "proton_pottsmpnn": ("ProtonPottsMPNN",),
    "proton-pottsmpnn": ("ProtonPottsMPNN",),
    "surfpro": ("SurfPro",), "carbondesign": ("CarbonDesign",),
    "dplm": ("DPLM",), "nampnn": ("NAMPNN",),
}


def _literal_names(path, decorator):
    """Built-in registry fallback; loaded runtime registrations take precedence."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == decorator and node.args:
            try:
                names.add(str(ast.literal_eval(node.args[0])).strip().lower())
            except (ValueError, TypeError):
                pass
    return names


def _expression_names(expression):
    return {node.id for node in ast.walk(ast.parse(expression, mode="eval")) if isinstance(node, ast.Name)}


def _check_expression(expression, path, errors):
    if expression is None:
        return set()
    try:
        return _expression_names(str(expression))
    except SyntaxError as exc:
        errors.append(f"{path}: {exc.msg}")
        return set()


@lru_cache(maxsize=1)
def _source_loss_optional():
    optional = {}
    for path in (Path(__file__).parent / "losses").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute) or dec.func.attr != "register":
                    continue
                values = {}
                for kw in dec.keywords:
                    try:
                        values[kw.arg] = ast.literal_eval(kw.value)
                    except (ValueError, TypeError):
                        pass
                if not {"params", "required_params", "optional_params"} & values.keys():
                    positional = node.args.posonlyargs + node.args.args
                    defaulted = {arg.arg for arg in positional[len(positional) - len(node.args.defaults):]} if node.args.defaults else set()
                    defaulted.update(arg.arg for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults) if default is not None)
                    values["optional_params"] = tuple(defaulted - {"eval_result", "runtime_config", "oligo2complex"})
                optional[str(values.get("name", node.name))] = set(values.get("optional_params", ()))
    return optional


def _check_selector(expression, path, errors):
    if expression in (None, "None"):
        return
    names = _check_expression(expression, path, errors)
    from astrolabe.selectors.registry import SelectorRegistry
    for name in names - set(SelectorRegistry.list_available()):
        errors.append(f"{path}: unknown selector {name!r}")


class StartupValidationError(ValueError):
    """One or more actionable startup errors."""

_VALIDATED_CONFIG = ContextVar("astrolabe_validated_startup_config", default=None)

def _config_revision(cfg):
    return deepcopy(OmegaConf.to_container(cfg, resolve=False)
                    if OmegaConf.is_config(cfg) else dict(cfg)) if isinstance(cfg, Mapping) else None


@contextmanager
def startup_validation_scope(cfg, *, require_inputs=False):
    if isinstance(cfg, Mapping):
        validate_startup_config(cfg, require_inputs=require_inputs)
    previous = _VALIDATED_CONFIG.get()
    revision = _config_revision(cfg)
    input_requirement = require_inputs or (
        previous is not None and previous[0] is cfg and previous[1]
        and previous[2] == revision
    )
    token = _VALIDATED_CONFIG.set((cfg, input_requirement, revision))
    try:
        yield
    finally:
        _VALIDATED_CONFIG.reset(token)


@lru_cache(maxsize=1)
def _source_specs():
    """Read registration metadata without importing PyRosetta/model frameworks."""
    root = Path(__file__).parent
    specs = {"evaluator": {}, "loss": {}}
    for family, paths in (
        ("evaluator", [root / "evaluators" / "utils.py"]),
        ("loss", sorted((root / "losses").glob("*.py"))),
    ):
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for decorator in node.decorator_list:
                    if not isinstance(decorator, ast.Call):
                        continue
                    func = decorator.func
                    if family == "evaluator":
                        if not isinstance(func, ast.Name) or func.id != "register_evaluator_factory":
                            continue
                        name = ast.literal_eval(decorator.args[0])
                        required = ()
                    else:
                        if not (
                            isinstance(func, ast.Attribute) and func.attr == "register"
                            and isinstance(func.value, ast.Name) and func.value.id == "loss_registry"
                        ):
                            continue
                        options = {}
                        for keyword in decorator.keywords:
                            try:
                                options[keyword.arg] = ast.literal_eval(keyword.value)
                            except (ValueError, TypeError):
                                pass
                        name = options.get("name", node.name)
                        required = options.get("required_params", ())
                        if not {"params", "required_params", "optional_params"} & options.keys():
                            positional = node.args.posonlyargs + node.args.args
                            required_names = {arg.arg for arg in positional[:len(positional) - len(node.args.defaults)]}
                            required_names.update(arg.arg for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults) if default is None)
                            required = tuple(required_names - {"eval_result", "runtime_config", "oligo2complex"})
                    key = str(name).lower() if family == "evaluator" else str(name)
                    specs[family][key] = tuple(required)
    # Programmatic RNA registration has literal backend descriptors.
    rna_source = root / "evaluators" / "rna.py"
    if rna_source.is_file():
        for node in ast.walk(ast.parse(rna_source.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "RnaBackend" and node.args
                    and isinstance(node.args[0], ast.Constant)):
                specs["evaluator"][node.args[0].value] = ()
    return specs


def validate_startup_config(cfg, *, check_dependencies=True, resolve_environment=True, require_inputs=False):
    """Validate selected configuration, leaving unused backend defaults unresolved.

    Returns resolved active loss terms for dependency checks. No output directory,
    model owner, random state or global scientific configuration is changed.
    """
    if not isinstance(cfg, Mapping):
        raise StartupValidationError("config: expected a mapping")
    validated = _VALIDATED_CONFIG.get()
    if (validated is not None and validated[0] is cfg
            and (validated[1] or not require_inputs)
            and validated[2] == _config_revision(cfg)):
        return {}
    errors = []
    raw = OmegaConf.to_container(cfg, resolve=False) if OmegaConf.is_config(cfg) else dict(cfg)
    if not isinstance(raw, dict):
        raise StartupValidationError("config: expected a mapping")

    reference_info = {}
    errors.extend(validate_task_config(
        cfg, raw=raw, resolve_environment=resolve_environment,
        require_inputs=require_inputs, error_type=StartupValidationError,
        reference_info=reference_info,
    ))

    def resolve(value, path):
        try:
            if OmegaConf.is_config(value):
                return OmegaConf.to_container(
                    value, resolve=resolve_environment, throw_on_missing=resolve_environment,
                )
            if value == "???":
                raise ValueError("mandatory value is missing")
            return value
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            return None

    terms_by_family = {}
    for family in ("evaluator", "mutator", "loss", "predocker"):
        section = raw.get(family)
        if section is None:
            section = {}
        if not isinstance(section, dict):
            errors.append(f"{family}: expected a mapping")
            terms_by_family[family] = {}
            continue
        terms = section.get(f"{family}_terms") or {}
        if not isinstance(terms, dict):
            errors.append(f"{family}.{family}_terms: expected an alias-to-config mapping")
            terms = {}
        terms_by_family[family] = terms

    stages = raw.get("stages") or {}
    if not isinstance(stages, dict):
        errors.append("stages: expected a mapping")
        stages = {}
    elif stages and OmegaConf.is_config(cfg):
        # Stage aliases and expressions use the same resolved values as StageManager.
        resolved_stages = {}
        for name in stages:
            if stages[name] is None:
                continue
            value = resolve(cfg.stages[name], f"stages.{name}")
            if value is not None:
                resolved_stages[name] = value
        stages = resolved_stages
    if stages and "default" not in stages:
        errors.append("stages.default: required default stage")
    selected = {"evaluator": set(), "mutator": set()}
    for name, stage in stages.items():
        if stage is None:
            continue
        if not isinstance(stage, dict):
            errors.append(f"stages.{name}: expected a mapping")
            continue
        for key in stage.keys() - _STAGE_FIELDS:
            errors.append(f"stages.{name}.{key}: unknown stage field")
        if name != "default" and not stage.get("_entrance"):
            errors.append(f"stages.{name}._entrance: required for nondefault stage")
        if name == "default":
            for key in ("evaluator", "mutators", "weighted_score_expression", "mutation_num"):
                if not stage.get(key) and stage.get(key) != 0:
                    errors.append(f"stages.default.{key}: required")
        if "mutation_num" in stage:
            try:
                value = cfg.stages[name].get("mutation_num") if OmegaConf.is_config(cfg) else stage["mutation_num"]
                if value is not None:
                    resolve_mutation_num(value, protomer_pool={})
            except Exception as exc:
                errors.append(f"stages.{name}.mutation_num: {exc}")
        for family in selected:
            stage_key = "mutators" if family == "mutator" else family
            aliases = stage.get(stage_key)
            aliases = aliases if isinstance(aliases, list) else [aliases]
            for alias in aliases:
                if alias is not None:
                    if not isinstance(alias, str):
                        errors.append(f"stages.{name}.{stage_key}: expected an alias string")
                    else:
                        selected[family].add(alias)
        gate = stage.get("transition_consistency_gate") or {}
        if isinstance(gate, dict) and gate.get("evaluator"):
            selected["evaluator"].add(str(gate["evaluator"]))
        for key in _STAGE_EXPRESSIONS & stage.keys():
            _check_expression(stage[key], f"stages.{name}.{key}", errors)

    from astrolabe.runtime_usage import _used_mutator_aliases
    selected["mutator"].update(_used_mutator_aliases({**raw, "stages": stages}))
    for family, aliases in selected.items():
        section = raw.get(family) or {}
        enabled = section.get("enabled_aliases") if isinstance(section, dict) else None
        if enabled is not None:
            if not isinstance(enabled, list) or not all(isinstance(a, str) for a in enabled):
                errors.append(f"{family}.enabled_aliases: expected a list of aliases")
            else:
                for alias in aliases - set(enabled):
                    errors.append(f"{family}.enabled_aliases: stage-selected alias {alias!r} is disabled")
                aliases.update(enabled)
        for alias in sorted(aliases):
            if terms_by_family[family].get(alias) is None:
                errors.append(f"{family}.{family}_terms.{alias}: selected alias is undefined or disabled")

    predocker_section = raw.get("predocker") if isinstance(raw.get("predocker"), dict) else {}
    predocker_enabled = predocker_section.get("enabled_aliases")
    if predocker_enabled is not None:
        if not isinstance(predocker_enabled, list) or not all(isinstance(alias, str) for alias in predocker_enabled):
            errors.append("predocker.enabled_aliases: expected a list of aliases")
        else:
            for alias in predocker_enabled:
                if terms_by_family["predocker"].get(alias) is None:
                    errors.append(f"predocker.predocker_terms.{alias}: selected alias is undefined or disabled")

    source_specs = _source_specs()
    evaluator_names = set(source_specs["evaluator"])
    registry_module = sys.modules.get("astrolabe.evaluators.registry")
    if registry_module is not None:
        evaluator_names.update(registry_module.EvaluatorRegistry.list_available())
    loss_names = set(source_specs["loss"])
    losses_module = sys.modules.get("astrolabe.losses.base")
    if losses_module is not None:
        loss_names.update(losses_module.loss_registry.valid_losses)
    mutator_names = _literal_names(Path(__file__).parent / "mutators" / "base.py", "register_mutator_factory")
    mutator_module = sys.modules.get("astrolabe.mutators.registry")
    if mutator_module is not None:
        mutator_names.update(mutator_module.MutatorRegistry.list_available())
    processor_names = set()
    processor_source = (Path(__file__).parent / "structure_processors" / "base.py").read_text(encoding="utf-8")
    processor_tree = ast.parse(processor_source)
    for node in processor_tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "_PREDOCKER_TYPE_TO_MODULE":
            processor_names.update(ast.literal_eval(node.value))
    processor_module = sys.modules.get("astrolabe.structure_processors.base")
    if processor_module is not None:
        processor_names.update(processor_module.StructureProcessorRegistry.list_available())
    active_losses = {}
    active_evaluators = []
    active_mutators = []
    active_loss_names = set()
    default_stage = stages.get("default") or {}
    all_losses = not bool(stages)
    for stage in stages.values():
        if not isinstance(stage, dict):
            continue
        active = stage.get("active_loss_terms")
        if active is None:
            all_losses = True
        elif isinstance(active, list):
            active_loss_names.update(active)
        else:
            errors.append("stages.*.active_loss_terms: expected a list or null")
    for alias in sorted(active_loss_names):
        if alias not in terms_by_family["loss"] and alias not in loss_names:
            errors.append(f"stages.*.active_loss_terms: unknown loss alias {alias!r}")
    for family, terms in terms_by_family.items():
        for alias, term in terms.items():
            if family == "predocker":
                enabled_predockers = predocker_section.get("enabled_aliases")
                if enabled_predockers is not None and alias not in enabled_predockers:
                    continue
            if term is None:
                if family == "predocker":
                    errors.append(f"predocker.predocker_terms.{alias}: null processor is not constructible")
                continue
            if family in selected and selected[family] and alias not in selected[family]:
                continue
            if family == "loss" and not all_losses and alias not in active_loss_names:
                continue
            path = f"{family}.{family}_terms.{alias}"
            if not isinstance(term, dict):
                errors.append(f"{path}: expected a mapping")
                continue
            if not isinstance(term.get("_type"), str) or not term["_type"].strip():
                errors.append(f"{path}._type: expected a non-empty component type")
                continue
            try:
                section_cfg = cfg[family]
                selected_term_cfg = section_cfg[f"{family}_terms"][alias]
                selected_kind = str(selected_term_cfg.get("_type", "")).strip()
                if family == "evaluator":
                    raw_kind = selected_kind.lower()
                    merged_cfg = merge_component_config(section_cfg, (raw_kind,), selected_term_cfg, resolve=resolve_environment)
                    resolved = OmegaConf.to_container(merged_cfg, resolve=resolve_environment)
                elif family == "mutator":
                    raw_kind = selected_kind.lower()
                    merged_cfg = merge_component_config(section_cfg, _MUTATOR_DEFAULTS.get(raw_kind, ()), selected_term_cfg, resolve=resolve_environment)
                    resolved = OmegaConf.to_container(merged_cfg, resolve=resolve_environment)
                elif family == "predocker":
                    raw_kind = selected_kind
                    canonical = "haddock3_nucleic" if raw_kind == "HADDOCK3_nucleic" else raw_kind
                    keys = ("haddock3", canonical, raw_kind) if canonical == "haddock3_nucleic" else (canonical, raw_kind)
                    merged_cfg = merge_component_config(section_cfg, keys, selected_term_cfg, resolve=resolve_environment)
                    resolved = OmegaConf.to_container(merged_cfg, resolve=resolve_environment)
                else:
                    resolved = resolve(selected_term_cfg, path)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                continue
            if not isinstance(resolved, dict):
                continue
            kind = resolved.get("_type")
            if not isinstance(kind, str) or not kind.strip():
                errors.append(f"{path}._type: expected a non-empty component type")
                continue
            if family == "evaluator":
                kind = kind.strip().lower()
                if kind not in evaluator_names:
                    errors.append(f"{path}._type: unknown evaluator {kind!r}")
                active_evaluators.append((alias, kind, None, resolved))
                from astrolabe.model_execution import execution_mode
                try:
                    mode, legacy = resolved.get("execution_mode"), resolved.get("execution")
                    value = mode if mode is not None else legacy
                    if not resolve_environment and isinstance(value, str) and "${" in value:
                        continue
                    if kind != "bioemu2":
                        execution_mode(value)
                    if kind == "alphafold2" and execution_mode(value).value == "in_process":
                        resource_config = resolve(cfg.get("execution", {}), "execution")
                        resource_mode = (resource_config or {}).get("mode", "legacy")
                        if (resource_mode in {"resident_serial", "adaptive"}
                                and not os.environ.get("ASTROLABE_RESOURCE_SOCKET")):
                            # GPU allocators and observed RSS remain charged until
                            # this process exits. Its own broker cannot drain first.
                            raise ValueError(
                                "in_process AlphaFold2 requires execution.mode=legacy "
                                "or an externally owned resource session; use "
                                "execution_mode=daemon with a local resource owner"
                            )
                    if kind != "bioemu2" and mode is not None and legacy is not None and execution_mode(mode) != execution_mode(legacy):
                        raise ValueError("execution_mode and execution disagree")
                except ValueError as exc:
                    errors.append(f"{path}: {exc}")
            if family == "mutator":
                active_mutators.append((alias, kind, resolved))
                if kind.strip().lower() not in mutator_names:
                    errors.append(f"{path}._type: unknown mutator {kind!r}")
                chain = resolved.get("predocker")
                if chain is not None:
                    if isinstance(chain, str):
                        chain = [chain]
                    if not isinstance(chain, list) or not all(isinstance(a, str) for a in chain):
                        errors.append(f"{path}.predocker: expected a list of aliases")
                    else:
                        for ref in chain:
                            if terms_by_family["predocker"].get(ref) is None:
                                errors.append(f"{path}.predocker: unknown predocker alias {ref!r}")
                            elif predocker_enabled is not None and isinstance(predocker_enabled, list) and ref not in predocker_enabled:
                                errors.append(f"{path}.predocker: disabled predocker alias {ref!r}")
                selector = resolved.get("position_selector")
                if selector is not None:
                    _check_selector(selector, f"{path}.position_selector", errors)
            if family == "predocker" and kind not in processor_names:
                errors.append(f"{path}._type: unknown predocker {kind!r}")
            if family in {"loss", "predocker"}:
                errors.extend(validate_scoring_settings(
                    family, kind, resolved, path=path,
                    check_paths=bool(resolve_environment and not raw.get("dry_run")),
                ))
            if family == "loss":
                active_losses[alias] = resolved
                if kind not in loss_names:
                    errors.append(f"{path}._type: unknown loss {kind!r}")
                metadata = None
                if losses_module is not None and kind in losses_module.loss_registry.valid_losses:
                    metadata = losses_module.loss_registry.get_metadata(kind)
                required = set(metadata.required_params or ()) if metadata is not None else set(source_specs["loss"].get(kind, ()))
                optional = set(metadata.optional_params or ()) if metadata is not None else _source_loss_optional().get(kind, set())
                for key in required:
                    if key not in resolved or resolved[key] is None:
                        errors.append(f"{path}.{key}: required by loss {kind!r}")
                if metadata is not None or kind in source_specs["loss"]:
                    for key in set(resolved) - {"_type", "_for"} - INTRINSIC_PARAMS - required - optional:
                        errors.append(f"{path}.{key}: invalid parameter for loss {kind!r}")

    from astrolabe.task_references import sequence_inventory, validate_loss_references
    # Alias/complex membership needs no model or structure-coordinate runtime.
    seqs = resolve(cfg.get("seqs", {}), "seqs") or {}
    types = resolve(cfg.get("types", {}), "types") or {}
    files = resolve(cfg.get("named_files", {}), "named_files") or {}
    if all(isinstance(v, dict) for v in (seqs, types, files)):
        def reference_error(path, message):
            errors.append(f"{path}: {message}")
        segments = reference_info.get("segments")
        if segments is None:
            _, segments, _ = sequence_inventory(seqs, types, files, check_paths=False,
                                                fail=reference_error)
        oligos = {
            oligo.lstrip("-~")
            for stage in stages.values() if isinstance(stage, dict)
            for oligo in stage.get("oligo") or [] if isinstance(oligo, str)
        }
        for alias, options in active_losses.items():
            validate_loss_references(
                options, path=f"loss.loss_terms.{alias}", oligos=oligos,
                segments=segments, fail=reference_error)

    loss_section = raw.get("loss") if isinstance(raw.get("loss"), dict) else {}
    expression_terms = loss_section.get("expression_terms") or {}
    if isinstance(expression_terms, dict):
        dependencies = {}
        expression_values = {}
        for name in expression_terms:
            try:
                expression = resolve(cfg.loss.expression_terms[name], f"loss.expression_terms.{name}")
                expression_values[name] = expression
                dependencies[name] = _check_expression(expression, f"loss.expression_terms.{name}", errors) & expression_terms.keys()
            except (AttributeError, KeyError) as exc:
                errors.append(f"loss.expression_terms.{name}: {exc}")
        visiting, visited = set(), set()
        def visit(name):
            if name in visiting:
                errors.append(f"loss.expression_terms.{name}: cyclic dependency")
                return
            if name in visited:
                return
            visiting.add(name)
            for dep in dependencies.get(name, ()):
                visit(dep)
            visiting.remove(name)
            visited.add(name)
        requested_expressions = set()
        for stage in stages.values():
            if isinstance(stage, dict):
                for key in _STAGE_EXPRESSIONS & stage.keys():
                    if stage[key] is not None:
                        requested_expressions.update(_check_expression(stage[key], f"stages.*.{key}", errors))
        for name in requested_expressions & dependencies.keys():
            visit(name)
        for stage_name, stage in stages.items():
            if not isinstance(stage, dict) or stage.get("active_loss_terms") is None:
                continue
            required = set()
            for key in _STAGE_EXPRESSIONS & stage.keys():
                required.update(_check_expression(stage[key], f"stages.{stage_name}.{key}", errors))
            pending = list(required & expression_terms.keys())
            while pending:
                name = pending.pop()
                for dep in dependencies.get(name, ()):
                    if dep not in required:
                        required.add(dep)
                        pending.append(dep)
                required.update(_check_expression(expression_values.get(name), f"loss.expression_terms.{name}", errors))
            omitted = (required & (set(terms_by_family["loss"]) | loss_names)) - set(stage["active_loss_terms"])
            if omitted:
                errors.append(f"stages.{stage_name}.active_loss_terms: expression dependencies omitted {sorted(omitted)}")
    else:
        errors.append("loss.expression_terms: expected a mapping")

    mutator_section = raw.get("mutator") if isinstance(raw.get("mutator"), dict) else {}
    selector_expression = mutator_section.get("selector_expression")
    if selector_expression is not None:
        selector_expression = resolve(cfg.mutator.selector_expression, "mutator.selector_expression")
        if isinstance(selector_expression, str):
            pieces = selector_expression.split(";") if ":" in selector_expression else [selector_expression]
            for piece in pieces:
                if ":" in selector_expression:
                    alias, _, expression = piece.partition(":")
                    if alias.strip() in selected["mutator"]:
                        _check_selector(expression.strip(), f"mutator.selector_expression.{alias.strip()}", errors)
                else:
                    _check_selector(piece, "mutator.selector_expression", errors)
        else:
            errors.append("mutator.selector_expression: expected a string")

    initializer = raw.get("initializer")
    if initializer is not None:
        if not isinstance(initializer, dict):
            errors.append("initializer: expected a mapping")
        else:
            try:
                initializer = resolve(cfg.initializer, "initializer")
                mode = str(initializer.get("mode") or "random").strip().lower()
                mode = {
                    "none": "random", "default": "random", "aa-set": "aa_set",
                    "aaset": "aa_set", "mask-mix": "mask_mix", "maskmix": "mask_mix",
                    "all-a": "all_a", "alla": "all_a", "x-mix": "x_mix",
                    "xmix": "x_mix", "rfd-3": "rfd3",
                    "start-from-structure": "structure", "start_from_structure": "structure",
                    "existing-structure": "structure", "existing_structure": "structure",
                }.get(mode, mode)
                initializer["mode"] = mode
                modes = {"random", "dplm", "aa_set", "mask_mix", "all_a", "pssm", "x_mix", "structure", "rfd3"}
                registry_module = sys.modules.get("astrolabe.initializers.registry")
                if registry_module is not None:
                    modes.update(registry_module.InitializerRegistry.list_available())
                if mode not in modes:
                    errors.append(f"initializer.mode: unknown initializer {mode!r}")
                if mode == "dplm" and initializer.get("dplm_alias") is not None:
                    dplm_alias = initializer["dplm_alias"]
                    if dplm_alias not in terms_by_family["mutator"]:
                        errors.append(f"initializer.dplm_alias: unavailable mutator alias {dplm_alias!r}")
                    elif str((terms_by_family["mutator"].get(dplm_alias) or {}).get("_type", "")).strip().lower() != "dplm":
                        errors.append(f"initializer.dplm_alias: {dplm_alias!r} is not a DPLM mutator")
                if mode == "aa_set" and initializer.get("aa_set") is None:
                    errors.append("initializer.aa_set: required for aa_set mode")
                if mode == "pssm" and initializer.get("pssm") is None:
                    errors.append("initializer.pssm: required for pssm mode")
                if mode in {"aa_set", "mask_mix", "x_mix"} and initializer.get("aa_set") is not None:
                    residues = initializer["aa_set"]
                    if not isinstance(residues, (str, list, dict)) or not residues:
                        errors.append("initializer.aa_set: expected non-empty string, list, or mapping")
                    elif isinstance(residues, dict) and mode == "aa_set" and not ({"residues", "segments"} & residues.keys()):
                        errors.append("initializer.aa_set: mapping requires residues and/or segments")
                if mode == "pssm" and initializer.get("pssm") is not None and not isinstance(initializer["pssm"], (str, dict)):
                    errors.append("initializer.pssm: expected path or mapping")
                if mode in {"mask_mix", "x_mix"}:
                    field = "mask_fraction" if mode == "mask_mix" else "x_fraction"
                    fraction = initializer.get(field, None if mode == "mask_mix" else 0.8)
                    try:
                        value = float(fraction)
                        if not math.isfinite(value) or not 0 <= value <= 1:
                            raise ValueError()
                    except (ValueError, TypeError):
                        errors.append(f"initializer.{field}: expected number within [0, 1]")
                if mode in {"structure", "rfd3"}:
                    designer = initializer.get("sequence_designer")
                    if designer not in selected["mutator"]:
                        errors.append(f"initializer.sequence_designer: unavailable mutator alias {designer!r}")
                    sub = initializer.get(mode)
                    if not isinstance(sub, dict):
                        errors.append(f"initializer.{mode}: expected a mapping")
                    else:
                        chain_map = sub.get("chain_map")
                        if not isinstance(chain_map, dict) or not chain_map:
                            errors.append(f"initializer.{mode}.chain_map: expected non-empty mapping")
                        elif any(not str(chain).strip() or not (isinstance(entry, str) and entry.strip() or isinstance(entry, dict) and str(entry.get("protomer") or entry.get("target") or "").strip()) for chain, entry in chain_map.items()):
                            errors.append(f"initializer.{mode}.chain_map: invalid chain or protomer entry")
                        if mode == "structure" and not sub.get("source"):
                            errors.append("initializer.structure.source: required")
                        if mode == "rfd3":
                            for key, minimum in (("n_batches", 1), ("sample_index", 0)):
                                value = sub.get(key, minimum)
                                if type(value) is not int or value < minimum:
                                    errors.append(f"initializer.rfd3.{key}: expected integer >= {minimum}")
            except Exception as exc:
                errors.append(f"initializer: {exc}")

    if raw.get("structure_start") is not None:
        from astrolabe.structure_start_config import StructureStartConfig
        try:
            structure_start = StructureStartConfig.from_raw(resolve(cfg.structure_start, "structure_start"))
            if structure_start is not None and structure_start.hotspot_gate.enabled:
                _check_expression(structure_start.hotspot_gate.expression, "structure_start.hotspot_gate.expression", errors)
        except ValueError as exc:
            errors.append(str(exc))

    from astrolabe.structure_only_startup import validate_structure_only_scores
    errors.extend(validate_structure_only_scores(
        active_evaluators=active_evaluators, active_mutators=active_mutators,
        active_losses=active_losses,
        per_residue_scores=resolve(cfg.get("per_residue_scores", {}), "per_residue_scores"),
        stages=stages, expression_terms=expression_terms,
    ))

    check_paths = bool(resolve_environment and not raw.get("dry_run"))
    for alias, kind, _unused, options in active_evaluators:
        errors.extend(validate_backend_settings(
            "evaluator", kind, options,
            path=f"evaluator.evaluator_terms.{alias}", check_paths=check_paths,
            allow_unresolved=not resolve_environment,
        ))
    for alias, kind, options in active_mutators:
        errors.extend(validate_backend_settings(
            "mutator", kind, options,
            path=f"mutator.mutator_terms.{alias}", check_paths=check_paths,
            allow_unresolved=not resolve_environment,
        ))
    if isinstance(initializer, dict) and initializer.get("mode") == "rfd3" and isinstance(initializer.get("rfd3"), dict):
        errors.extend(validate_backend_settings(
            "initializer", "rfd3", initializer["rfd3"],
            path="initializer.rfd3", check_paths=check_paths,
            allow_unresolved=not resolve_environment,
        ))

    for key, minimum in (("steps", 0), ("repeats", 1)):
        if key not in cfg:
            continue
        try:
            value = resolve(cfg[key], key)
            if type(value) is not int or value < minimum:
                errors.append(f"{key}: expected an integer >= {minimum}")
        except Exception as exc:
            errors.append(f"{key}: {exc}")
    if "seed" in cfg:
        try:
            seed = cfg["seed"]
            if seed is not None and (type(seed) is not int or not 1 <= seed <= 2**32 - 1):
                errors.append("seed: expected null or an integer in [1, 2**32 - 1]")
        except Exception as exc:
            errors.append(f"seed: {exc}")

    from astrolabe.configuration import validate_execution_selection
    from astrolabe.evaluator_startup import initialization_options, validate_independent_plan
    try:
        validate_execution_selection(cfg)
        _workers, initialization_seed = initialization_options(cfg.get("evaluator", {}))
        validate_independent_plan(active_evaluators, initialization_seed)
    except (ValueError, TypeError) as exc:
        errors.append(str(exc))
    if errors:
        raise StartupValidationError("Startup configuration errors:\n- " + "\n- ".join(errors))
    if check_dependencies:
        from astrolabe.startup_dependencies import (
            check_core_runtime_dependencies, check_startup_dependencies,
        )
        check_startup_dependencies({
            **raw, "loss": {**(raw.get("loss") or {}), "loss_terms": active_losses},
        })
        if require_inputs:
            check_core_runtime_dependencies()
    return active_losses
