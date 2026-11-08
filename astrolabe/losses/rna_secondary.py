"""CPU secondary-structure scores from current RNA sequences.

Targets are ordered mappings of name to ``strands`` (protomer occurrences),
``structure`` (dot-parenthesis-plus), and optional nonnegative ``weight``.
Scores are weighted means across complexes, not tube yields. Defect is already
normalized and is minimized; structure probability is maximized; ensemble free
energy is in kcal/mol. Use a negative defect in a maximizing score expression.

The calculate functions take physical sequence strings instead of protomer names.
NUPACK defaults are material='rna', celsius=37, sodium=1 M, magnesium=0 M,
ensemble='stacking'. RNA salt corrections are unavailable. The explicit
backend='viennarna' supports single strands with native ensemble settings only;
omit sodium/magnesium and use ensemble=None or 'vienna'. Neither backend is
imported until a calculation is requested.

Public APIs: https://docs.nupack.org/utilities/ and https://docs.nupack.org/model/;
https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/api/python.html.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import ProtomerTypes
from astrolabe.losses.base import loss_registry


_SETTINGS = ["backend", "material", "celsius", "sodium", "magnesium", "ensemble"]
_METRICS = {"ensemble_defect", "structure_probability", "ensemble_free_energy"}


@dataclass(frozen=True)
class _Target:
    strands: tuple[str, ...]
    structure: str
    weight: float


def _finite_number(value, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _validate_strands(strands) -> tuple[str, ...]:
    if isinstance(strands, (str, bytes)) or not isinstance(strands, Sequence) or not strands:
        raise ValueError("strands must be a non-empty ordered sequence of RNA strings.")
    result = []
    for sequence in strands:
        if not isinstance(sequence, str) or not sequence or set(sequence.upper()) - set("ACGU"):
            raise ValueError("Each RNA strand must be a non-empty ACGU sequence.")
        result.append(sequence.upper())
    return tuple(result)


def _validate_structure(strands: tuple[str, ...], structure: str) -> None:
    if not isinstance(structure, str) or not structure or set(structure) - set(".()+"):
        raise ValueError("structure must use dot-parenthesis-plus notation without pseudoknots.")
    parts = structure.split("+")
    if len(parts) != len(strands) or any(len(part) != len(seq) for part, seq in zip(parts, strands)):
        raise ValueError("structure strand lengths and separators must match the ordered strands.")
    depth = 0
    for symbol in structure:
        depth += (symbol == "(") - (symbol == ")")
        if depth < 0:
            raise ValueError("structure parentheses must be balanced.")
    if depth:
        raise ValueError("structure parentheses must be balanced.")


def _resolve_targets(targets) -> tuple[_Target, ...]:
    if not isinstance(targets, Mapping) or not targets:
        raise ValueError("targets must be a non-empty ordered mapping of target names to specifications.")
    resolved = []
    for name, spec in targets.items():
        if not isinstance(name, str) or not name or not isinstance(spec, Mapping):
            raise ValueError("Each target needs a non-empty name and a mapping specification.")
        if set(spec) - {"strands", "structure", "weight"} or not {"strands", "structure"} <= set(spec):
            raise ValueError(f"Target {name!r} requires strands and structure, with only optional weight.")
        occurrences = spec["strands"]
        if isinstance(occurrences, (str, bytes)) or not isinstance(occurrences, Sequence) or not occurrences:
            raise ValueError(f"Target {name!r} strands must be an ordered list of protomer names.")
        sequences = []
        for protomer_name in occurrences:
            if not isinstance(protomer_name, str) or protomer_name not in global_config.protomers:
                raise ValueError(f"Target {name!r} has unknown RNA protomer {protomer_name!r}.")
            protomer = global_config.protomers[protomer_name]
            if getattr(protomer, "type", None) is not ProtomerTypes.RNA:
                raise ValueError(f"Target {name!r} requires an RNA protomer, got {protomer_name!r}.")
            sequence = getattr(protomer, "mutated_sequence", None)
            if not isinstance(sequence, str) or not sequence:
                raise ValueError(f"RNA protomer {protomer_name!r} requires a non-empty mutated_sequence.")
            sequences.append(sequence)
        strands = _validate_strands(sequences)
        structure = spec["structure"]
        _validate_structure(strands, structure)
        weight = _finite_number(spec.get("weight", 1.0), f"Target {name!r} weight")
        if weight < 0:
            raise ValueError("Target weight must be nonnegative.")
        resolved.append(_Target(strands, structure, weight))
    if not any(target.weight > 0 for target in resolved):
        raise ValueError("targets must have a positive total weight.")
    return tuple(resolved)


def calculate_rna_secondary(
    strands,
    structure=None,
    *,
    metric="ensemble_defect",
    backend="nupack",
    material="rna",
    celsius=37.0,
    sodium=None,
    magnesium=None,
    ensemble=None,
) -> float:
    """Calculate one complex score without an evaluator or global configuration.

    ``strands`` contains actual sequences in physical occurrence order. A target
    structure is required except for ensemble_free_energy. Explicit unsupported
    settings raise ValueError before an optional dependency is imported.
    """
    strands = _validate_strands(strands)
    if not isinstance(metric, str) or metric not in _METRICS:
        raise ValueError(f"Unknown RNA secondary metric {metric!r}.")
    if structure is not None or metric != "ensemble_free_energy":
        _validate_structure(strands, structure)
    if material != "rna":
        raise ValueError("RNA secondary scoring requires material='rna'.")
    temperature = _finite_number(celsius, "celsius")
    if temperature <= -273.15:
        raise ValueError("celsius must exceed absolute zero.")

    if backend == "nupack":
        sodium = 1.0 if sodium is None else _finite_number(sodium, "sodium")
        magnesium = 0.0 if magnesium is None else _finite_number(magnesium, "magnesium")
        if sodium != 1.0 or magnesium != 0.0:
            raise ValueError("NUPACK RNA supports only sodium=1.0 M and magnesium=0.0 M; RNA salt corrections are unavailable.")
        ensemble = "stacking" if ensemble is None else ensemble
        if ensemble not in ("stacking", "dangle-stacking", "coaxial-stacking", "nostacking"):
            raise ValueError(f"Unsupported NUPACK ensemble {ensemble!r}.")
        module_name = "nupack"
    elif backend == "viennarna":
        if len(strands) != 1:
            raise ValueError("The viennarna backend supports a single strand only.")
        if sodium is not None or magnesium is not None:
            raise ValueError("The viennarna backend does not support explicit sodium or magnesium settings.")
        if ensemble not in (None, "vienna"):
            raise ValueError("The viennarna backend requires its native ensemble, not a NUPACK ensemble.")
        module_name = "RNA"
    else:
        raise ValueError(f"Unknown RNA secondary backend {backend!r}; choose 'nupack' or 'viennarna'.")

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"RNA secondary scoring backend {backend!r} requires the optional {module_name!r} Python dependency. "
            "Provide an existing authorized installation; no backend fallback is performed."
        ) from exc

    if backend == "nupack":
        model = module.Model(
            material=material, celsius=temperature, sodium=sodium,
            magnesium=magnesium, ensemble=ensemble,
        )
        if metric == "ensemble_free_energy":
            value = module.pfunc(strands=list(strands), model=model)[1]
        else:
            method = module.defect if metric == "ensemble_defect" else module.structure_probability
            value = method(strands=list(strands), structure=structure, model=model)
    else:
        model = module.md()
        model.temperature = temperature
        model.compute_bpp = 1
        compound = module.fold_compound(strands[0], model)
        _, mfe = compound.mfe()
        compound.exp_params_rescale(mfe)
        _, free_energy = compound.pf()
        if metric == "ensemble_free_energy":
            value = free_energy
        elif metric == "ensemble_defect":
            value = compound.ensemble_defect(structure)
        else:
            value = compound.pr_structure(structure)

    score = _finite_number(value, f"{backend} {metric} result")
    if metric != "ensemble_free_energy" and not 0.0 <= score <= 1.0:
        raise ValueError(f"{backend} {metric} result must be in [0, 1], got {score}.")
    return score


def calculate_rna_ensemble_defect(strands, structure, **settings) -> float:
    """Return normalized complex ensemble defect, lower is better."""
    return calculate_rna_secondary(strands, structure, metric="ensemble_defect", **settings)


def calculate_rna_structure_probability(strands, structure, **settings) -> float:
    """Return equilibrium target-structure probability in [0, 1]."""
    return calculate_rna_secondary(strands, structure, metric="structure_probability", **settings)


def calculate_rna_ensemble_free_energy(strands, **settings) -> float:
    """Return complex ensemble free energy in kcal/mol, not MFE structure energy."""
    return calculate_rna_secondary(strands, metric="ensemble_free_energy", **settings)


def _weighted_score(targets, metric, **settings) -> float:
    resolved = _resolve_targets(targets)
    total = math.fsum(target.weight for target in resolved)
    return math.fsum(
        target.weight * calculate_rna_secondary(
            target.strands, target.structure, metric=metric, **settings,
        )
        for target in resolved if target.weight > 0
    ) / total


@loss_registry.register(required_params=["targets"], optional_params=_SETTINGS)
def rna_ensemble_defect(eval_result, targets, backend="nupack", **settings):
    """Weighted normalized complex defect from current mutated RNA sequences."""
    return _weighted_score(targets, "ensemble_defect", backend=backend, **settings)


@loss_registry.register(required_params=["targets"], optional_params=_SETTINGS[1:])
def nupack_ensemble_defect(eval_result, targets, **settings):
    """Explicit NUPACK-only alias of rna_ensemble_defect."""
    return _weighted_score(targets, "ensemble_defect", backend="nupack", **settings)


@loss_registry.register(required_params=["targets"], optional_params=_SETTINGS)
def rna_structure_probability(eval_result, targets, backend="nupack", **settings):
    """Weighted target-structure probability, higher is better."""
    return _weighted_score(targets, "structure_probability", backend=backend, **settings)


@loss_registry.register(required_params=["targets"], optional_params=_SETTINGS)
def rna_ensemble_free_energy(eval_result, targets, backend="nupack", **settings):
    """Weighted complex ensemble free energy in kcal/mol."""
    return _weighted_score(targets, "ensemble_free_energy", backend=backend, **settings)
