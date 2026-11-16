"""Bounded caches for deterministic input preparation, never model outputs."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import functools
import hashlib
from pathlib import Path
import threading

import numpy as np

from astrolabe.utils.af3_ccd import create_isolated_ccd


class MsaRowCache:
    """Reuse encoded alignment rows while native MSA assembly stays fresh.

    Keys include the exact aligned string (including insertions), polymer type
    and encoder. Query validation, deduplication, pairing, species identifiers,
    profile calculation and cropping remain in the native pipeline. Returned
    arrays own their storage; no caller can modify a stored row.
    """

    def __init__(self, *, max_entries=32768, max_bytes=64 * 1024 * 1024):
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("Cache limits must be non-negative")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._rows = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self.hits = 0
        self.encoded_rows = 0

    def clear(self):
        with self._lock:
            self._rows.clear()
            self._bytes = 0

    def extract(self, encoder, msa_sequences, chain_poly_type):
        # Keep the native validation/errors for unusual callers and empty MSAs.
        if (not isinstance(msa_sequences, (list, tuple)) or not msa_sequences
                or not all(isinstance(row, str) for row in msa_sequences)
                or not self.max_entries or not self.max_bytes):
            return encoder(msa_sequences, chain_poly_type)
        with self._lock:
            keys = [(encoder, chain_poly_type, row) for row in msa_sequences]
            missing = list(dict.fromkeys(key for key in keys if key not in self._rows))
            resolved = {key: self._rows[key][:2] for key in keys if key in self._rows}
            fresh = {}
            if missing:
                try:
                    encoded, deletions = encoder([key[2] for key in missing], chain_poly_type)
                except ValueError:
                    # Preserve native errors, including the original row index.
                    return encoder(msa_sequences, chain_poly_type)
                if (not isinstance(encoded, np.ndarray) or not isinstance(deletions, np.ndarray)
                        or encoded.ndim != 2 or encoded.shape != deletions.shape
                        or encoded.shape[0] != len(missing)):
                    return encoder(msa_sequences, chain_poly_type)
                for index, key in enumerate(missing):
                    row = encoded[index].copy()
                    deletion = deletions[index].copy()
                    row.setflags(write=False)
                    deletion.setflags(write=False)
                    fresh[key] = (row, deletion)
                resolved.update(fresh)
            lengths = {resolved[key][0].shape for key in keys}
            if len(lengths) != 1:
                # A new query length must not reuse an incompatible old MSA.
                return encoder(msa_sequences, chain_poly_type)
            self.hits += sum(key in self._rows for key in keys)
            self.encoded_rows += len(missing)
            for key in keys:
                if key in self._rows:
                    self._rows.move_to_end(key)
            for key, (row, deletion) in fresh.items():
                size = row.nbytes + deletion.nbytes + len(key[2].encode("utf-8"))
                if size > self.max_bytes:
                    continue
                self._rows[key] = (row, deletion, size)
                self._bytes += size
                while len(self._rows) > self.max_entries or self._bytes > self.max_bytes:
                    _, (_, _, removed_size) = self._rows.popitem(last=False)
                    self._bytes -= removed_size
            return (np.stack([resolved[key][0] for key in keys]),
                    np.stack([resolved[key][1] for key in keys]))


_MSA_CACHE = ContextVar("astrolabe_msa_row_cache", default=None)
_INSTALL_LOCK = threading.Lock()


def install_msa_row_dispatch(msa_features):
    """Install a stateless dispatcher; each request carries its own cache."""
    with _INSTALL_LOCK:
        original = msa_features.extract_msa_features
        if getattr(original, "_astrolabe_msa_rows", False):
            return

        @functools.wraps(original)
        def extract_msa_features(msa_sequences, chain_poly_type):
            cache = _MSA_CACHE.get()
            if cache is None:
                return original(msa_sequences, chain_poly_type)
            return cache.extract(original, msa_sequences, chain_poly_type)

        extract_msa_features._astrolabe_msa_rows = True
        msa_features.extract_msa_features = extract_msa_features


@contextmanager
def msa_row_cache_scope(cache):
    token = _MSA_CACHE.set(cache)
    try:
        yield
    finally:
        _MSA_CACHE.reset(token)


class AF3CcdCache:
    """Reuse one isolated CCD; custom entries never mutate the native base.

    The pinned AF3 fork keys reference conformers by CCD entry identity. Clear
    those references before releasing an overlay so Python id reuse cannot
    attach an old conformer to a different custom component. The caller holds
    the shared AF3 featurisation lock while accessing or clearing this owner.
    """

    def __init__(self):
        self._entry = None
        self._references = None

    def get(self, chemical_components, features, user_ccd):
        key = (chemical_components.Ccd, user_ccd)
        if self._entry is not None and self._entry[:2] == key:
            return self._entry[2]
        self.clear()
        self._references = getattr(features, "_REFSTRUCTURE_REFERENCE_CACHE", None)
        ccd = create_isolated_ccd(chemical_components, user_ccd=user_ccd)
        if user_ccd is not None and self._references is not None:
            self._references.clear()
        self._entry = (*key, ccd)
        return ccd

    def clear(self):
        if (self._entry is not None and self._entry[1] is not None
                and self._references is not None):
            self._references.clear()
        self._entry = None


@contextmanager
def af3_ccd_scope(cache, chemical_components, features, user_ccd):
    """Release temporary custom CCD references before the preparation lock exits.

    Evaluator-owned CCDs survive requests until explicit owner shutdown. Static
    prediction callers have no such owner, so their overlay must be released on
    both success and failure while the shared featurisation lock is still held.
    """
    owner = cache if cache is not None else AF3CcdCache()
    try:
        yield owner.get(chemical_components, features, user_ccd)
    finally:
        if cache is None:
            owner.clear()


class FileObjectCache:
    """One owner loads an unchanged asset once, with file-version validation.

    Metadata is only a fast path within an owner lifetime. In-place writes and
    atomic replacements invalidate through ctime/inode as well as mtime/size.
    A loader failure or a concurrently changed file is never published.
    """

    def __init__(self):
        self._entry = None
        self._lock = threading.RLock()

    @staticmethod
    def _identity(path):
        stat = path.stat()
        return (str(path), stat.st_dev, stat.st_ino, stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns)

    def load(self, path, loader):
        path = Path(path).resolve(strict=True)
        with self._lock:
            identity = self._identity(path)
            if self._entry is not None and self._entry[:2] == (identity, loader):
                return self._entry[2]
            value = loader(path)
            if identity != self._identity(path):
                raise RuntimeError(f"Preparation asset changed while loading: {path}")
            self._entry = (identity, loader, value)
            return value

    def clear(self):
        with self._lock:
            self._entry = None


class CopyOnAccessMapping(MutableMapping):
    """Request-local lazy copies of a loaded molecular dictionary.

    Native parsers can mutate RDKit molecules. Copy only the requested entries,
    never the entire CCD, and keep the owner's asset isolated between requests.
    """

    def __init__(self, source):
        self._source = source
        self._copies = {}
        self._deleted = set()

    def __getitem__(self, key):
        if key in self._deleted:
            raise KeyError(key)
        if key not in self._copies:
            self._copies[key] = copy.deepcopy(self._source[key])
        return self._copies[key]

    def __setitem__(self, key, value):
        self._deleted.discard(key)
        self._copies[key] = value

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(key)
        self._copies.pop(key, None)
        self._deleted.add(key)

    def __iter__(self):
        yield from (key for key in self._source if key not in self._deleted)
        yield from (key for key in self._copies if key not in self._source)

    def __len__(self):
        return sum(1 for _ in self)

    def __contains__(self, key):
        return key not in self._deleted and (key in self._copies or key in self._source)


class ContentCache:
    """Small owner-local exact-content cache with defensive copies."""

    def __init__(self, max_entries=16):
        self.max_entries = max_entries
        self._entries = OrderedDict()

    def clear(self):
        self._entries.clear()

    def get_or_create(self, text, factory):
        key = (factory, hashlib.sha256(text.encode("utf-8")).digest())
        if key not in self._entries:
            value = factory(text)
            self._entries[key] = copy.deepcopy(value)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return value
        self._entries.move_to_end(key)
        return copy.deepcopy(self._entries[key])
