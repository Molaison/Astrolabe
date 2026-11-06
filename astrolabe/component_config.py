"""Merge a selected component without resolving an overridden default."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from omegaconf import DictConfig, OmegaConf, open_dict, read_write


def merge_component_config(section, default_keys, override, *, resolve=True) -> DictConfig:
    """Return an independent config, resolving only after defaults and alias merge.

    A temporary child of a copy of the original root retains absolute OmegaConf
    interpolations such as ``${paths.model}`` during the final resolution.
    """
    if not OmegaConf.is_config(section) and not isinstance(section, Mapping):
        section = {
            name: getattr(section, name) for name in dir(section)
            if not name.startswith("_") and not callable(getattr(section, name))
        }
    keys = (default_keys,) if isinstance(default_keys, str) else tuple(default_keys)
    layers = []
    def disposable_layer(value):
        raw = OmegaConf.to_container(value, resolve=False) if OmegaConf.is_config(value) else value
        return OmegaConf.create(raw)
    for key in keys:
        if key in section and section[key] is not None:
            layers.append(disposable_layer(section[key]))
    if override is not None:
        layers.append(disposable_layer(override))
    merged = OmegaConf.merge(*layers) if layers else OmegaConf.create({})
    if not OmegaConf.is_dict(merged):
        raise ValueError("component configuration must be a mapping")
    parent = section if OmegaConf.is_config(section) else OmegaConf.create(section)
    chain = []
    while parent._get_parent() is not None:
        chain.append(parent._key())
        parent = parent._get_parent()
    root = copy.deepcopy(parent)
    target = root
    for key in reversed(chain):
        target = target[key]
    selected_key = "__selected_component_config__"
    selected_parent = target
    if OmegaConf.is_config(override) and override._get_parent() is not None:
        original_parent = override._get_parent()
        if original_parent._get_parent() is section:
            selected_parent = target[original_parent._key()]
            selected_key = override._key()
    # The root is disposable; relax inherited flags only there. Never alter the
    # caller's read-only/struct flags or source nodes.
    with read_write(root), open_dict(root), read_write(selected_parent), open_dict(selected_parent):
        selected_parent[selected_key] = merged
        selected = selected_parent[selected_key]
        if not resolve:
            return selected
        value = OmegaConf.to_container(selected, resolve=True, throw_on_missing=True)
    return OmegaConf.create(value)
