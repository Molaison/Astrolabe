"""Preserve OmegaConf merging while identifying container-shape errors."""
from collections.abc import Mapping

from omegaconf import OmegaConf
from omegaconf.errors import ConfigTypeError


def _merge_shape(left, right, path):
    if isinstance(left, Mapping) and isinstance(right, list) or (
        isinstance(left, list) and isinstance(right, Mapping)
    ):
        raise ValueError(f"{path or 'config'}: cannot merge a mapping with a list")
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result = dict(left)
        for key, value in right.items():
            child_path = f"{path}.{key}" if path else str(key)
            result[key] = _merge_shape(result[key], value, child_path) if key in result else value
        return result
    return right


def merge_configs(*layers, path=""):
    """Merge normally; only inspect unresolved shapes when OmegaConf fails."""
    try:
        return OmegaConf.merge(*layers)
    except ConfigTypeError:
        shape = None
        for layer in layers:
            raw = OmegaConf.to_container(layer, resolve=False) if OmegaConf.is_config(layer) else layer
            shape = _merge_shape(shape, raw, path)
        raise
