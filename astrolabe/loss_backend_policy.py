"""Lightweight dependency policy shared by scoring, startup and recovery gates."""
from __future__ import annotations

import ast
import importlib.util
from collections.abc import Mapping


INDEPENDENT_LOSS_IMPORTS = {
    "protomer_shape_complementarity": ("biotite", "shape_complementarity"),
    "ligand_shape_complementarity": ("biotite", "shape_complementarity"),
    "interface_shape_complementarity": ("biotite", "shape_complementarity"),
    "contact_molecular_surface": ("biotite", "py_contact_ms", "scipy"),
    "contact_molecular_surface_norm": ("biotite", "py_contact_ms", "scipy"),
    "interface_hbond_count": ("biotite", "hydride"),
    "ligand_protein_hbond": ("biotite", "hydride"),
    "interface_buried_unsatisfied_hbonds": ("biotite", "hydride"),
}

NATIVE_STRUCTURAL_LOSSES = frozenset({
    "interface_holes", "chain_holes_diff", "ligand_local_holes",
    "ligand_buried_unsat_hbonds", "interface_polar_satisfaction",
    "short_range_interface_pair_energy",
})


def independent_loss_imports(name: str, params: Mapping | None = None) -> tuple[str, ...]:
    imports = INDEPENDENT_LOSS_IMPORTS.get(name, ())
    if (params or {}).get("hydrogen_policy") == "require":
        return tuple(module for module in imports if module != "hydride")
    return imports


def selector_requires_rosetta(expression: str | None) -> bool:
    """Consult selector ownership without importing selector implementations."""
    if not expression or expression == "None":
        return False
    from astrolabe.selectors.registry import SelectorRegistry

    tree = ast.parse(expression, mode="eval")
    variables = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    native = SelectorRegistry.pose_required_variables() - SelectorRegistry.path_only_variables()
    return bool(variables & native)


def loss_selectors_require_rosetta(params: Mapping | None = None) -> bool:
    return any(selector_requires_rosetta((params or {}).get(field)) for field in (
        "position_selector_a", "position_selector_b", "position_selector", "select_expression",
    ))


def loss_requires_rosetta(name: str, params: Mapping | None = None) -> bool | None:
    """Resolve selectable backends without importing any scientific package.

    None leaves unrelated losses with their existing dependency owner. Auto
    can still reject unsupported chemistry at runtime; availability is not
    evidence of equivalence between the independent and Rosetta definitions.
    """
    if name in NATIVE_STRUCTURAL_LOSSES:
        if (params or {}).get("backend", "auto") == "independent":
            raise ValueError(f"{name} has no validated independent backend; use auto or rosetta")
        return True
    imports = independent_loss_imports(name, params)
    if not imports:
        return None
    params = params or {}
    backend = params.get("backend", "auto")
    if backend == "rosetta":
        return True
    if backend == "independent":
        return False
    if backend != "auto":
        raise ValueError(f"Unsupported loss backend {backend!r}: use auto, independent or rosetta")
    if name in {"contact_molecular_surface", "contact_molecular_surface_norm"}:
        if params.get("use_rosetta_radii") is True:
            return True
    if name == "interface_buried_unsatisfied_hbonds":
        if params.get("ignore_surface_res", False) or params.get("separation_distance", 1000.0) != 1000.0:
            return True
    if name in {"interface_shape_complementarity", "interface_hbond_count"}:
        if any(params.get(key) is not None for key in (
            "hbond_energy_cutoff", "interface_distance", "buried_sasa_cutoff", "sasa_probe_radius",
        )):
            return True
    return any(importlib.util.find_spec(module) is None for module in imports)
