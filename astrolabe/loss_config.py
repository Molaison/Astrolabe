"""Lightweight loss settings shared by preflight and runtime consumers."""
import ast
from collections.abc import Mapping, Sequence

INTRINSIC_PARAMS = frozenset({"oligomer_specific"})

# The first entry is the deterministic default, never an availability fallback.
PHYSICAL_SCORE_BACKENDS = {
    "protomer_shape_complementarity": ("sc_rs", "rosetta"),
    "interface_shape_complementarity": ("sc_rs", "rosetta"),
    "ligand_shape_complementarity": ("sc_rs", "rosetta"),
    "contact_molecular_surface": ("py_contact_ms", "rosetta"),
    "contact_molecular_surface_norm": ("py_contact_ms", "rosetta"),
    "interface_hbond_count": ("biotite", "rosetta"),
    "interface_buried_unsatisfied_hbonds": ("biotite", "rosetta"),
    "ligand_protein_hbond": ("biotite", "rosetta"),
    "ligand_buried_unsat_hbonds": ("biotite", "rosetta"),
    "interface_holes": ("rosetta",),
    "chain_holes_diff": ("rosetta",),
    "ligand_local_holes": ("rosetta",),
}


def loss_backend(kind, options=None):
    """Resolve a physical backend without importing its implementation."""
    choices = PHYSICAL_SCORE_BACKENDS.get(kind)
    if choices is None:
        return None
    backend = (options or {}).get("backend", choices[0])
    if not isinstance(backend, str) or backend not in choices:
        raise ValueError(f"backend for {kind!r} must be one of {', '.join(choices)}")
    return backend


def normalize_loss_options(options: Mapping) -> dict:
    """Normalize oligomer scope and reject conflicting compatibility fields."""
    result = dict(options)
    if result.get("_type") in {"interface_hbond_count", "interface_shape_complementarity"}:
        for common, native in (("protomer_a", "binder_chains"), ("protomer_b", "target_chains")):
            if common not in result:
                continue
            if native in result:
                raise ValueError(f"{common} conflicts with legacy {native}; use one form")
            result[native] = result.pop(common)

    def scope(value):
        values = value.split(",") if isinstance(value, str) else value
        if not isinstance(values, Sequence) or not values or any(
            not isinstance(item, str) or not item.strip() for item in values
        ):
            raise ValueError("_for/oligomer_specific must be a nonempty oligomer string or list")
        return list(dict.fromkeys(item.strip() for item in values))
    explicit = result.pop("_for", None)
    legacy = result.pop("oligomer_specific", None)
    for_scope = scope(explicit) if explicit is not None else None
    old_scope = scope(legacy) if legacy is not None else None
    if for_scope is not None and old_scope is not None and set(for_scope) != set(old_scope):
        raise ValueError("_for and oligomer_specific specify conflicting oligomers")
    selected = for_scope if for_scope is not None else old_scope
    if selected is not None:
        result["oligomer_specific"] = selected
    return result


def validate_loss_selector(expression):
    """Validate the existing selector language without loading selector classes."""
    if expression is None:
        return
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expected a nonempty positional selector expression or null")
    from astrolabe.selectors.registry import SelectorRegistry
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid positional selector syntax: {exc.msg}") from exc
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Call, ast.Name,
        ast.Load, ast.Constant, ast.keyword, ast.List, ast.Tuple, ast.Dict,
        ast.Add, ast.Sub, ast.BitAnd, ast.BitOr, ast.BitXor, ast.Invert,
        ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    )
    available = set(SelectorRegistry.list_available())
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"unsupported positional selector syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in available:
            raise ValueError(f"unknown positional selector {node.id!r}")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValueError("positional selector calls must name a registered selector")
        if isinstance(node, ast.keyword) and node.arg is None:
            raise ValueError("positional selector keyword expansion is unsupported")
    if not isinstance(tree.body, (ast.Call, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Constant)):
        raise ValueError("positional selector must return a selector or boolean")
    if isinstance(tree.body, ast.Constant):
        value = tree.body.value
        if value is not None and not isinstance(value, bool):
            raise ValueError("positional selector must return a selector or boolean")
