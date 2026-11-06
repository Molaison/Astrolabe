"""Configuration-selected scientific factories.

The catalog stores owner-qualified module targets.  Routing imports a selected
target only at execution time and contains no task or protocol branches.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping


SearchFactory = Callable[[Any, Any], Any]

SEARCH_FACTORY_CATALOG: dict[tuple[str, str], str | SearchFactory] = {
    (
        "branching",
        "isolated_native",
    ): "astrolabe.protein_beam:run_configured_branching",
    ("branching", "native_refinement"):
        "astrolabe.design.configured_refinement:run_configured_refinement",
    (
        "branching",
        "archived_refinement",
    ): "astrolabe.rna_beam:run_configured_refinement",
}

# Migration guard only: never use this set to infer an execution factory.
_LEGACY_RNA_MEASUREMENTS = frozenset({"confidence", "bivalent_geometry"})
_PROTEIN_BEAM_EXECUTION_KEYS = frozenset({
    "af3_call_budget",
    "case_input",
    "gates",
    "holes_ranking",
    "initial_state",
    "initial_state_audit",
    "max_evaluations",
    "max_generations",
    "max_total_mutations",
    "pipeline_proposals",
    "proposal_batch_size",
    "proposals_per_parent",
    "root_calibration",
    "root_parity_reference",
    "selection",
    "service",
    "width",
})
_MISSING = object()


def register_search_factory(
    algorithm: str,
    owner: str,
    target: str | SearchFactory,
    *,
    replace: bool = False,
) -> None:
    """Register one ``(algorithm, owner)`` implementation."""

    key = (str(algorithm).strip(), str(owner).strip())
    if not all(key):
        raise ValueError("search algorithm and owner must be non-empty")
    if key in SEARCH_FACTORY_CATALOG and not replace:
        raise ValueError(f"Search factory already registered for {key!r}")
    if not isinstance(target, str):
        if not callable(target):
            raise TypeError("search factory target must be module:function or callable")
    SEARCH_FACTORY_CATALOG[key] = target


def _target_callable(target: str) -> SearchFactory:
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            f"Search factory target must use module:function, got {target!r}"
        )
    value = importlib.import_module(module_name)
    for part in attribute.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"Search factory target {target!r} is not callable")
    return value


def load_composed_config(path: str | Path) -> Any:
    """Load a standalone or Hydra-composed task configuration.

    Standalone files overlay ``config/base.yaml``. Files with a ``defaults``
    list compose from their own directory, with packaged repository configs on
    the Hydra search path. This is the shared loader used by configuration
    owners and model services; it does not select an execution algorithm.
    """

    from omegaconf import OmegaConf, open_dict

    from astrolabe import config_dir
    from astrolabe.cli import _compose_hydra_config

    config_path = Path(path).expanduser().resolve()
    repository_config = Path(config_dir()).expanduser().resolve()
    standalone = OmegaConf.load(config_path)
    if "defaults" in standalone:
        searchpath = json.dumps([repository_config.as_uri()])
        return _compose_hydra_config(
            config_dir=config_path.parent,
            config_name=config_path.stem,
            overrides=["env=env", f"hydra.searchpath={searchpath}"],
        )

    base = _compose_hydra_config(
        config_dir=repository_config,
        config_name="base",
        overrides=["env=env"],
    )
    with open_dict(base):
        return OmegaConf.merge(base, standalone)


def selected_search_target(cfg: Any) -> tuple[str, str] | None:
    """Return one fully explicit owner-qualified search selection."""

    if cfg is None or not hasattr(cfg, "get"):
        return None
    if "search" not in cfg:
        return None
    search = cfg.get("search")
    if search is None:
        return None
    if not isinstance(search, Mapping):
        raise TypeError("search configuration must be a mapping")
    if not search:
        raise ValueError("search configuration must select a non-empty factory")
    algorithm = _required_selection_text(search, "algorithm")
    owner = _required_selection_text(search, "owner")
    if not algorithm or not owner:
        raise ValueError("search algorithm and owner must be non-empty")
    return algorithm, owner


def _required_selection_text(search: Mapping, key: str) -> str:
    value = search.get(key, _MISSING)
    if value is _MISSING or value is None:
        raise ValueError(f"search.{key} is required")
    if not isinstance(value, str):
        raise TypeError(f"search.{key} must be a string")
    value = value.strip()
    if not value or value == "???" or "${" in value:
        raise ValueError(f"search.{key} must be explicitly configured")
    return value


def _mapping_field(cfg: Any, key: str) -> Mapping | None:
    value = cfg.get(key) if cfg is not None and hasattr(cfg, "get") else None
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _has_protein_beam_execution(cfg: Any) -> bool:
    protein_beam = _mapping_field(cfg, "protein_beam")
    return protein_beam is not None and bool(
        _PROTEIN_BEAM_EXECUTION_KEYS.intersection(protein_beam)
    )


def validate_execution_selection(cfg: Any) -> tuple[str, str] | None:
    """Validate trajectory/search routing before planning or model execution."""

    key = selected_search_target(cfg)
    beam_search = _mapping_field(cfg, "beam_search")
    if key is not None:
        if key not in SEARCH_FACTORY_CATALOG:
            raise ValueError(f"Unknown selected search factory {key!r}")
        if beam_search:
            raise ValueError(
                "Legacy beam_search cannot be combined with explicit "
                "search.algorithm and search.owner"
            )
        if (
            key != ("branching", "isolated_native")
            and _has_protein_beam_execution(cfg)
        ):
            raise ValueError(
                "protein_beam execution settings conflict with selected "
                f"search owner {key[1]!r}; use search.owner='isolated_native'"
            )
        search = cfg.get("search")
        measurement = search.get("measurement")
        if (
            key == ("branching", "native_refinement")
            and isinstance(measurement, str)
            and measurement.strip().lower() in _LEGACY_RNA_MEASUREMENTS
        ):
            raise ValueError(
                "RNA measurement configs must migrate search.owner from "
                "'native_refinement' to 'archived_refinement'; refusing "
                "silent routing to the generic refinement adapter"
            )
        return key

    paper_protocol = _mapping_field(cfg, "paper_protocol")
    engine = None if paper_protocol is None else paper_protocol.get("engine")
    normalized_engine = (
        str(engine).strip().lower() if engine is not None else None
    )
    if normalized_engine not in {None, "trajectory"}:
        raise ValueError(
            f"paper_protocol.engine={engine!r} is deprecated; explicit "
            "search.algorithm and search.owner are required"
        )

    if beam_search:
        raise ValueError(
            "Legacy beam_search configuration has no execution route; "
            "migrate it to explicit search.algorithm and search.owner"
        )

    if _has_protein_beam_execution(cfg):
        raise ValueError(
            "Protein-beam execution settings require explicit "
            "search.algorithm and search.owner"
        )
    return None


def dispatch_selected_search(cfg: Any, repeat: Any) -> bool:
    """Run a selected search factory, returning whether search was handled.

    An unknown key or malformed catalog target raises.  There is no fallback to
    a different algorithm after selection.
    """

    key = validate_execution_selection(cfg)
    if key is None:
        return False
    search = cfg.get("search")
    summary_format = search.get("summary_format")
    if summary_format is not None:
        summary_format = str(summary_format).strip().lower()
        if summary_format != "json":
            raise ValueError(
                "search.summary_format must be null or 'json', "
                f"got {search.get('summary_format')!r}"
            )
    target = SEARCH_FACTORY_CATALOG.get(key)
    if target is None:
        raise ValueError(f"Unknown selected search factory {key!r}")
    factory = _target_callable(target) if isinstance(target, str) else target
    result = factory(cfg, repeat)
    if summary_format == "json":
        print(json.dumps(result, default=str, allow_nan=False))
    return True


__all__ = [
    "SEARCH_FACTORY_CATALOG",
    "dispatch_selected_search",
    "load_composed_config",
    "register_search_factory",
    "selected_search_target",
    "validate_execution_selection",
]
