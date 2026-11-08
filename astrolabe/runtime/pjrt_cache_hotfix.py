"""Opt-in, target-interpreter validation of the qualified CUDA12 cache plugin.

Only metadata and the final filesystem view are inspected; never import JAX or
load the shared object. Deployment supplies the read-only single-file bind.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from importlib import metadata
from importlib.machinery import PathFinder
from pathlib import Path
import sys

HOTFIX_SELECTOR = "jax-0.4.34-cuda12"
PLUGIN_SHA256 = "a2297ac0796d6961b24f9dc44131b529108ec684be421bb6a622efd1b6f87a25"
_DISTRIBUTIONS = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt")
_VERSION = "0.4.34"
_PLUGIN = Path("jax_plugins/xla_cuda12/xla_cuda_plugin.so")


@dataclass(frozen=True)
class VerifiedPJRT:
    plugin_path: Path
    sha256: str
    distributions: tuple[tuple[str, str], ...]


def validate_pjrt_hotfix(environment: Mapping[str, str]) -> VerifiedPJRT | None:
    selector = environment.get("ASTROLABE_PJRT_CACHE_HOTFIX")
    if selector is None:
        return None
    if selector != HOTFIX_SELECTOR:
        raise ValueError(f"Unknown ASTROLABE_PJRT_CACHE_HOTFIX: {selector!r}")
    try:
        installed = {name: metadata.distribution(name) for name in _DISTRIBUTIONS}
    except metadata.PackageNotFoundError as error:
        raise ValueError(f"PJRT hotfix requires the installed {_VERSION} CUDA12 stack: {error}") from error
    versions = tuple((name, dist.version) for name, dist in installed.items())
    if any(version != _VERSION for _, version in versions):
        raise ValueError(f"PJRT hotfix requires every distribution at {_VERSION}: {versions!r}")
    plugin = Path(installed["jax-cuda12-pjrt"].locate_file(_PLUGIN)).resolve()
    # Resolve plugin discovery without executing either package's __init__.
    namespace = PathFinder.find_spec("jax_plugins", sys.path)
    locations = namespace.submodule_search_locations if namespace is not None else None
    package = PathFinder.find_spec("jax_plugins.xla_cuda12", locations) if locations else None
    discovered = package.submodule_search_locations if package is not None else None
    if not discovered or {(Path(p) / _PLUGIN.name).resolve() for p in discovered} != {plugin}:
        raise ValueError(f"PJRT hotfix plugin discovery does not match installed distribution: {plugin}")
    try:
        with plugin.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as error:
        raise ValueError(f"PJRT hotfix cannot read mounted plugin {plugin}: {error}") from error
    if digest != PLUGIN_SHA256:
        raise ValueError(f"PJRT hotfix mounted plugin SHA256 mismatch: {plugin}: {digest}")
    return VerifiedPJRT(plugin, digest, versions)
