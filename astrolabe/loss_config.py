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


def loss_selectors_require_pose(options):
    """Use selector registrations to determine the structure context required."""
    from astrolabe.selectors.registry import SelectorRegistry
    pose_variables = SelectorRegistry.pose_required_variables()
    for field in ("selector_a", "selector_b", "position_selector"):
        expression = options.get(field)
        if not isinstance(expression, str) or not expression.strip():
            continue
        names = {node.id for node in ast.walk(ast.parse(expression.strip(), mode="eval"))
                 if isinstance(node, ast.Name)}
        if names & pose_variables:
            return True
    return False


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



def iter_selected_loss_configs(cfg):
    """Yield source/settings pairs selected by all reachable stage expressions.

    Explicit non-null terms run even if absent from an expression, unless a
    stage limits active_loss_terms. Expression aliases are traversed without
    importing losses.
    """
    loss = cfg.get("loss") or {}
    terms = loss.get("loss_terms") or {}
    expressions = loss.get("expression_terms") or {}
    stages = cfg.get("stages") or {}
    default = stages.get("default") or {}
    contexts = [(f"stages.{name}", stage) for name, stage in stages.items()
                if stage is not None] or [("", {})]
    fields = ("weighted_score_expression", "acceptance_threshold", "acceptance_gate",
              "early_stopping", "prefix_early_stopping", "_entrance")
    for stage_path, stage in contexts:
        active = stage.get("active_loss_terms")
        for alias, spec in terms.items():
            if spec is not None and (active is None or alias in active):
                yield f"loss.loss_terms.{alias}._type ({spec.get('_type')})", spec
        for name in active or ():
            if terms.get(name) is None:
                yield f"{stage_path}.active_loss_terms ({name})", {"_type": name}
        pending = []
        for field in fields:
            owner = stage if field in stage else default
            expression = owner.get(field)
            if isinstance(expression, str) and expression:
                owner_path = stage_path if field in stage else "stages.default"
                pending.append((f"{owner_path}.{field}", expression, field == "_entrance"))
        visited = set()
        while pending:
            field_path, expression, entrance = pending.pop()
            names = {node.id if isinstance(node, ast.Name) else node.attr
                     for node in ast.walk(ast.parse(expression, mode="eval"))
                     if isinstance(node, (ast.Name, ast.Attribute))}
            for name in sorted(names):
                if name in expressions and (name, entrance) not in visited:
                    visited.add((name, entrance))
                    pending.append((f"loss.expression_terms.{name}", expressions[name], entrance))
                elif entrance and terms.get(name) is not None:
                    # Entrance conditions use the source stage's engine, not
                    # the destination stage's active-loss whitelist.
                    yield f"{field_path} ({name})", terms[name]
                elif (name not in terms and name not in expressions
                      and (entrance or active is None or name in active)):
                    yield f"{field_path} ({name})", {"_type": name}
