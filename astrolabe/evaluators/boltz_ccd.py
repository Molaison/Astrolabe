"""Runner-owned Boltz CCD loading with request-local molecule copies."""
from __future__ import annotations
import copy
import pickle
from collections.abc import Mapping
from pathlib import Path


class _RequestComponents(Mapping):
    def __init__(self, components):
        self._components = components
        self._copies = {}

    def __getitem__(self, key):
        if key not in self._copies:
            self._copies[key] = copy.deepcopy(self._components[key])
        return self._copies[key]

    def __iter__(self):
        return iter(self._components)

    def __len__(self):
        return len(self._components)


class BoltzCcdCache:
    """Keep one asset per runner; mutations stay in a request's molecule copies."""

    def __init__(self):
        self._entry = None

    @staticmethod
    def _identity_for(path):
        stat = path.stat()
        return (str(path), stat.st_dev, stat.st_ino, stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns)

    def for_request(self, path):
        path = Path(path).expanduser().resolve()
        identity = self._identity_for(path)
        entry = self._entry
        if entry is None or identity != entry[0]:
            with path.open("rb") as stream:
                components = pickle.load(stream)
            if not isinstance(components, Mapping):
                raise ValueError("Boltz CCD asset must contain a component mapping")
            if self._identity_for(path) != identity:
                raise RuntimeError("Boltz CCD asset changed while it was being read")
            entry = (identity, components)
            self._entry = entry
        return _RequestComponents(entry[1])

    def clear(self):
        self._entry = None
