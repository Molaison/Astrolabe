"""Reuse deterministic AF3 preparation without retaining seeded model batches."""
from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import dataclasses
import functools
import hashlib
import inspect
import json
import os
import threading

import numpy as np


_ACTIVE_CACHE = ContextVar("astrolabe_af3_preparation_cache", default=None)
_INSTALL_LOCK = threading.Lock()
_PROTEIN_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _digest(parts):
    digest = hashlib.sha256()
    for part in parts:
        payload = part.encode("utf-8") if isinstance(part, str) else part
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.digest()


def _array_bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if dataclasses.is_dataclass(value):
        return sum(_array_bytes(getattr(value, f.name)) for f in dataclasses.fields(value))
    if isinstance(value, dict):
        return sum(_array_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_array_bytes(item) for item in value)
    return 0


def create_request_ccd(*, user_ccd=None, ccd_pickle_path=None):
    """Apply user chemistry to one request, preserving the native shared CCD."""
    from alphafold3.constants import chemical_components
    from alphafold3.cpp import cif_dict

    ccd = chemical_components.Ccd(ccd_pickle_path=ccd_pickle_path)
    if user_ccd is None:
        return ccd
    if not user_ccd:
        raise ValueError("User CCD cannot be an empty string.")
    user_entries = {
        key: {name: tuple(values) for name, values in entry.items()}
        for key, entry in cif_dict.parse_multi_data_cif(user_ccd).items()
    }
    # The pinned native constructor updates the dictionary returned by its
    # global pickle cache. Keep its native entry format, but detach this request.
    ccd._dict = {**ccd._dict, **user_entries}
    return ccd


class AF3PreparationCache:
    """One bounded, disposable cache per evaluator; callers never own cached arrays."""

    def __init__(self, *, max_bytes=128 * 1024 * 1024, max_entries=32):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("max_entries must be a nonnegative integer")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._bytes = 0
        self._counts = Counter()
        self._lock = threading.RLock()

    @property
    def statistics(self):
        with self._lock:
            return dict(self._counts, entries=len(self._entries), bytes=self._bytes)

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def _get(self, kind, key):
        with self._lock:
            entry = self._entries.get((kind, key))
            self._counts[f"{kind}_{'hits' if entry is not None else 'misses'}"] += 1
            if entry is not None:
                self._entries.move_to_end((kind, key))
                return entry[0]
        return None

    def _put(self, kind, key, value):
        size = _array_bytes(value)
        if not self.max_entries or not self.max_bytes or size > self.max_bytes:
            return
        stored = copy.deepcopy(value)
        with self._lock:
            old = self._entries.pop((kind, key), None)
            if old is not None:
                self._bytes -= old[1]
            self._entries[(kind, key)] = (stored, size)
            self._bytes += size
            while self._bytes > self.max_bytes or len(self._entries) > self.max_entries:
                _, (_, removed_size) = self._entries.popitem(last=False)
                self._bytes -= removed_size

    def extract_msa_features(self, original, *, msa_sequences, chain_poly_type):
        if len(msa_sequences) < 2 or not all(isinstance(row, str) for row in msa_sequences):
            return original(msa_sequences=msa_sequences, chain_poly_type=chain_poly_type)
        query = original(msa_sequences=msa_sequences[:1], chain_poly_type=chain_poly_type)
        key = _digest((chain_poly_type, *msa_sequences[1:]))
        tail = self._get("msa_tail", key)
        if tail is None:
            try:
                tail = original(msa_sequences=msa_sequences[1:], chain_poly_type=chain_poly_type)
            except ValueError:
                # Preserve native row numbers and the query in validation errors.
                return original(msa_sequences=msa_sequences, chain_poly_type=chain_poly_type)
            self._put("msa_tail", key, tail)
        if any(first.shape[1:] != rest.shape[1:] for first, rest in zip(query, tail, strict=True)):
            # Keep the native shape error, including the original query identity.
            return original(msa_sequences=msa_sequences, chain_poly_type=chain_poly_type)
        return tuple(np.concatenate((first, rest), axis=0)
                     for first, rest in zip(query, tail, strict=True))

    def template_features(self, original, arguments, *, protein_type):
        key = _template_key(arguments, protein_type=protein_type)
        if key is None:
            return original(**arguments)
        value = self._get("templates", key)
        if value is not None:
            return copy.deepcopy(value)
        value = original(**arguments)
        self._put("templates", key, value)
        return value


def _template_key(arguments, *, protein_type):
    expected = {"all_tokens", "standard_token_idxs", "padding_shapes",
                "fold_input", "max_templates", "logging_name"}
    if set(arguments) != expected:
        return None
    fold = arguments["fold_input"]
    if getattr(fold, "bonded_atom_pairs", None) or getattr(fold, "user_ccd", None):
        return None
    chains = tuple(fold.chains)
    if (not chains or any(not isinstance(chain, protein_type)
            or getattr(chain, "ptms", ()) or not chain.sequence
            or set(chain.sequence) - _PROTEIN_ALPHABET for chain in chains)):
        return None
    tokens = arguments["all_tokens"]
    padding = arguments["padding_shapes"]
    if not dataclasses.is_dataclass(padding):
        return None
    token_chains = np.asarray(tokens.chain_id)
    expected_chains = np.concatenate([np.repeat(chain.id, len(chain.sequence)) for chain in chains])
    if not np.array_equal(token_chains, expected_chains):
        return None
    # Native templates deduplicate by full residue-name sequence. Preserve that
    # partition: a mutation can merge or split entities even at constant length.
    entities = {}
    rows = []
    for chain in chains:
        residue_names = tuple(np.asarray(tokens.res_name)[token_chains == chain.id].tolist())
        entity = entities.setdefault(residue_names, len(entities))
        templates = []
        for template in chain.templates:
            if not isinstance(template.mmcif, str):
                return None
            templates.append((template.mmcif,
                              sorted(dict(template.query_to_template_map).items())))
        rows.append((chain.id, len(chain.sequence), entity, templates))
    metadata = {
        "chains": rows,
        "tokens": {field: np.asarray(getattr(tokens, field)).tolist()
                   for field in ("atom_name", "res_id", "chain_id", "chain_type")},
        "standard_token_idxs": np.asarray(arguments["standard_token_idxs"]).tolist(),
        "padding": dataclasses.asdict(padding),
        "max_templates": arguments["max_templates"],
    }
    return _digest((json.dumps(metadata, sort_keys=True, separators=(",", ":")),))


def _install_dispatch():
    from alphafold3.common.folding_input import ProteinChain
    from alphafold3.data import msa_features
    from alphafold3.model import features

    with _INSTALL_LOCK:
        original_msa = msa_features.extract_msa_features
        if not getattr(original_msa, "_astrolabe_preparation_dispatch", False):
            @functools.wraps(original_msa)
            def extract_msa_features(msa_sequences, chain_poly_type):
                active = _ACTIVE_CACHE.get()
                if active is None or not active[1]:
                    return original_msa(msa_sequences, chain_poly_type)
                return active[0].extract_msa_features(
                    original_msa, msa_sequences=msa_sequences, chain_poly_type=chain_poly_type)

            extract_msa_features._astrolabe_preparation_dispatch = True
            msa_features.extract_msa_features = extract_msa_features

        original_templates = features.Templates.compute_features
        if not getattr(original_templates, "_astrolabe_preparation_dispatch", False):
            signature = inspect.signature(original_templates)

            @functools.wraps(original_templates)
            def compute_features(cls, *args, **kwargs):
                active = _ACTIVE_CACHE.get()
                if active is None or cls is not features.Templates:
                    return original_templates.__func__(cls, *args, **kwargs)
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                return active[0].template_features(
                    original_templates, bound.arguments, protein_type=ProteinChain)

            compute_features._astrolabe_preparation_dispatch = True
            features.Templates.compute_features = classmethod(compute_features)


@contextmanager
def preparation_cache_scope(cache):
    """Activate request-local dispatch; unrelated native callers retain their path."""
    if cache is None:
        token = _ACTIVE_CACHE.set(None)
        try:
            yield
        finally:
            _ACTIVE_CACHE.reset(token)
        return
    _install_dispatch()
    msa_enabled = os.environ.get("ALPHAFOLD3_MSA_CACHE", "1").lower() not in {"0", "false", "no"}
    token = _ACTIVE_CACHE.set((cache, msa_enabled))
    try:
        yield
    finally:
        _ACTIVE_CACHE.reset(token)
