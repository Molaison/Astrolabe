"""Bridge between mutable legacy globals and the unified design contracts.

The legacy scientific state is process-global.  This module therefore exposes
one serialized owner boundary instead of presenting global state as though it
were independently threadable.  Translation deliberately keeps unsupported
legacy chemistry and ambiguous aggregate measurements visible as errors or
explicit legacy records rather than flattening them into invented identities.

This foundational bridge supports forward translation (legacy state to
System/Candidate/Evidence).  It does not claim a lossless inverse from typed
Evidence to a live ``StandardScores`` instance; that requires an explicit
persisted owner format and remains a separate capability.

Snapshots restore existing Python objects in the current process only. Live
Pose/process-resource handles require a separately qualified owner integration
and are rejected here, including when an unowned resource Blob is supplied.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import threading
from typing import Any, Literal

from astrolabe.core import global_config
from astrolabe.core.chain_resolution import chain_ids_for_oligo
from astrolabe.core.folding_input import Template

# Every bridge entry point shares one process-local scientific owner.
# This lock is not a process-resource registry or a cross-process Pose codec.
from .owner import SCIENTIFIC_OWNER as _LEGACY_OWNER
from .owner import scientific_session
from .store import ArrayCodec, DataStore, JSONCodec
from .types import (
    Artifact,
    Assignment,
    Blob,
    Candidate,
    Chemical,
    Dependency,
    Domain,
    Evidence,
    Key,
    Measurement,
    Occurrence,
    Polymer,
    Sample,
    Subject,
    System,
    key_for,
    make_candidate,
    make_system,
    manifest,
)

_CHEMICAL_CODEC = JSONCodec("astrolabe.legacy.chemical/v1")
_INDEX_MAP_CODEC = JSONCodec("astrolabe.legacy.index-map/v1")
_JSON = JSONCodec()

_TEMPLATE_FIELDS = ("mmcif", "query_to_template_map")
_TEMPLATE_REFERENCE = "$legacy_template/v1"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _decode_json(data: bytes) -> Any:
    return json.loads(
        data.decode("utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant {value!r}")
        ),
    )


def _content_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class LegacyTemplateCodec:
    """Explicit canonical codec for the legacy protein homology Template."""

    identity = "astrolabe.legacy.template/v1"

    def canonical_payload(self, value: Any) -> dict[str, Any]:
        if type(value) is not Template:
            raise UnsupportedLegacyState(
                "legacy template codec only accepts "
                "astrolabe.core.folding_input.Template"
            )
        observed_fields = tuple(field.name for field in fields(value))
        if observed_fields != _TEMPLATE_FIELDS:
            raise UnsupportedLegacyState(
                "legacy Template fields changed and require an explicit codec update: "
                f"{observed_fields!r}"
            )

        mmcif = value.mmcif
        if mmcif is None:
            mmcif_payload = None
        elif isinstance(mmcif, str):
            encoded = mmcif.encode("utf-8")
            mmcif_payload = {
                "content": mmcif,
                "dtype": "utf-8",
                "shape": [len(encoded)],
                "sha256": _content_sha256(encoded),
            }
        else:
            raise UnsupportedLegacyState(
                "legacy Template.mmcif must be a string or None"
            )

        mapping = value.query_to_template_map
        if not isinstance(mapping, Mapping):
            raise UnsupportedLegacyState(
                "legacy Template.query_to_template_map must be a mapping"
            )
        pairs = []
        for query, target in mapping.items():
            if (
                type(query) is not int
                or type(target) is not int
            ):
                raise UnsupportedLegacyState(
                    "legacy Template query/template indices must be integers"
                )
            pairs.append([query, target])
        pairs.sort(key=lambda item: item[0])
        return {
            "schema": self.identity,
            "template_type": (
                f"{Template.__module__}.{Template.__qualname__}"
            ),
            "fields": list(_TEMPLATE_FIELDS),
            "mmcif": mmcif_payload,
            "query_to_template_map": {
                "dtype": "python-int",
                "shape": [len(pairs), 2],
                "sha256": _content_sha256(_json_bytes(pairs)),
                "entries": pairs,
            },
        }

    def encode(self, value: Any) -> bytes:
        return _json_bytes(self.canonical_payload(value))

    def decode(self, data: bytes) -> Template:
        if not isinstance(data, bytes):
            raise TypeError("encoded legacy Template data must be bytes")
        value = _decode_json(data)
        if not isinstance(value, dict) or set(value) != {
            "schema",
            "template_type",
            "fields",
            "mmcif",
            "query_to_template_map",
        }:
            raise ValueError("invalid legacy Template payload")
        if value["schema"] != self.identity:
            raise ValueError("legacy Template schema mismatch")
        expected_type = f"{Template.__module__}.{Template.__qualname__}"
        if value["template_type"] != expected_type:
            raise ValueError("legacy Template type mismatch")
        if value["fields"] != list(_TEMPLATE_FIELDS):
            raise ValueError("legacy Template field list mismatch")

        mmcif_value = value["mmcif"]
        if mmcif_value is None:
            mmcif = None
        else:
            if not isinstance(mmcif_value, dict) or set(mmcif_value) != {
                "content",
                "dtype",
                "shape",
                "sha256",
            }:
                raise ValueError("invalid legacy Template mmCIF payload")
            mmcif = mmcif_value["content"]
            if not isinstance(mmcif, str) or mmcif_value["dtype"] != "utf-8":
                raise ValueError("invalid legacy Template mmCIF content")
            encoded = mmcif.encode("utf-8")
            if mmcif_value["shape"] != [len(encoded)]:
                raise ValueError("legacy Template mmCIF shape mismatch")
            if mmcif_value["sha256"] != _content_sha256(encoded):
                raise ValueError("legacy Template mmCIF content hash mismatch")

        mapping_value = value["query_to_template_map"]
        if not isinstance(mapping_value, dict) or set(mapping_value) != {
            "dtype",
            "shape",
            "sha256",
            "entries",
        }:
            raise ValueError("invalid legacy Template map payload")
        entries = mapping_value["entries"]
        if mapping_value["dtype"] != "python-int" or not isinstance(entries, list):
            raise ValueError("invalid legacy Template map dtype")
        if mapping_value["shape"] != [len(entries), 2]:
            raise ValueError("legacy Template map shape mismatch")
        if mapping_value["sha256"] != _content_sha256(_json_bytes(entries)):
            raise ValueError("legacy Template map content hash mismatch")
        query_to_template_map = {}
        for entry in entries:
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or type(entry[0]) is not int
                or type(entry[1]) is not int
            ):
                raise ValueError("legacy Template map entries must be integer pairs")
            if entry[0] in query_to_template_map:
                raise ValueError("legacy Template map contains duplicate query indices")
            query_to_template_map[entry[0]] = entry[1]
        return Template(
            mmcif=mmcif,
            query_to_template_map=query_to_template_map,
        )

    def references(self, value: Template) -> tuple[Blob, ...]:
        self.canonical_payload(value)
        return ()

    def compatibility_key(self) -> tuple[str, str]:
        return (type(self).__name__, self.identity)


_TEMPLATE_CODEC = LegacyTemplateCodec()


def _blob_record(blob: Blob) -> dict[str, str]:
    return {"codec": blob.codec, "digest": blob.identity.digest}


def _blob_from_record(value: Any) -> Blob:
    if not isinstance(value, Mapping) or set(value) != {"codec", "digest"}:
        raise UnsupportedLegacyState("invalid legacy Blob record")
    codec = value["codec"]
    digest = value["digest"]
    if not isinstance(codec, str) or not codec:
        raise UnsupportedLegacyState("invalid legacy Blob codec")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise UnsupportedLegacyState("invalid legacy Blob digest")
    return Blob(Key(digest), codec)


def _template_references(value: Any) -> list[Blob]:
    if isinstance(value, Mapping):
        if set(value) == {_TEMPLATE_REFERENCE}:
            reference = value[_TEMPLATE_REFERENCE]
            if not isinstance(reference, Mapping) or set(reference) != {
                "blob",
                "fields",
                "mmcif_sha256",
                "query_to_template_map_sha256",
            }:
                raise UnsupportedLegacyState("invalid legacy Template reference")
            if reference["fields"] != list(_TEMPLATE_FIELDS):
                raise UnsupportedLegacyState(
                    "legacy Template reference field list mismatch"
                )
            for name in ("mmcif_sha256", "query_to_template_map_sha256"):
                digest = reference[name]
                if name == "mmcif_sha256" and digest is None:
                    continue
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise UnsupportedLegacyState(
                        f"invalid legacy Template {name}"
                    )
            return [_blob_from_record(reference.get("blob"))]
        result = []
        for item in value.values():
            result.extend(_template_references(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_template_references(item))
        return result
    return []


def decode_legacy_templates(value: Any, *, data: DataStore) -> Any:
    """Recursively reconstruct explicit Template references in legacy metadata."""

    if not isinstance(data, DataStore):
        raise TypeError("data must be a DataStore")
    if isinstance(value, Mapping):
        if set(value) == {_TEMPLATE_REFERENCE}:
            reference = value[_TEMPLATE_REFERENCE]
            blobs = _template_references(value)
            if len(blobs) != 1:
                raise UnsupportedLegacyState("invalid legacy Template reference")
            blob = blobs[0]
            template = data.get(blob, _TEMPLATE_CODEC)
            payload = _TEMPLATE_CODEC.canonical_payload(template)
            mmcif = payload["mmcif"]
            if reference.get("fields") != payload["fields"]:
                raise UnsupportedLegacyState("legacy Template field reference mismatch")
            if reference.get("mmcif_sha256") != (
                None if mmcif is None else mmcif["sha256"]
            ):
                raise UnsupportedLegacyState("legacy Template mmCIF reference mismatch")
            if reference.get("query_to_template_map_sha256") != payload[
                "query_to_template_map"
            ]["sha256"]:
                raise UnsupportedLegacyState(
                    "legacy Template query-map reference mismatch"
                )
            return template
        return {
            str(key): decode_legacy_templates(item, data=data)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [decode_legacy_templates(item, data=data) for item in value]
    return value


class _LegacyTopologyCodec(JSONCodec):
    """Topology JSON that closes over canonical legacy Template blobs."""

    def references(self, value: Any) -> tuple[Blob, ...]:
        super().references(value)
        return tuple(_template_references(value))

    def compatibility_key(self) -> tuple[str, str]:
        # Existing readers use JSONCodec with this same identity.
        return (JSONCodec.__name__, self.identity)


_TOPOLOGY_CODEC = _LegacyTopologyCodec("astrolabe.legacy.topology/v1")

_SNAPSHOT_CONSTANTS = frozenset(
    {
        "binding_site",
        "chai_restraints",
        "covalent_bonds",
        "cyclization",
        "dynamic_binding_site",
        "dynamic_binding_site_enabled",
        "enable_pyrosetta",
        "initial_temperature",
        "ligand_resnames",
        "nonstandard_aliases",
        "num_tokens",
        "out_dir",
        "per_residue_scores",
        "protenix_constraint",
        "skip_complex_poses",
        "structure_start",
        "use_metropolis_acceptance",
        "user_ccd",
    }
)
_PROTOCOL = key_for(("astrolabe.design.legacy/v1",))


class UnsupportedLegacyState(ValueError):
    """Legacy state cannot be represented without losing scientific meaning."""


class LegacyFileBytesCodec:
    """Owner codec for immutable bytes read from a legacy structure file."""

    identity = "astrolabe.legacy.file-bytes/v1"

    def encode(self, value: bytes) -> bytes:
        if not isinstance(value, bytes):
            raise TypeError("legacy artifact content must be bytes")
        return value

    def decode(self, data: bytes) -> bytes:
        if not isinstance(data, bytes):
            raise TypeError("encoded legacy artifact content must be bytes")
        return bytes(data)

    def references(self, value: bytes) -> tuple[Blob, ...]:
        if not isinstance(value, bytes):
            raise TypeError("legacy artifact content must be bytes")
        return ()

    def compatibility_key(self) -> tuple[str, str]:
        return (type(self).__name__, self.identity)


_FILE_BYTES_CODEC = LegacyFileBytesCodec()
LEGACY_REFERENCE_CODECS = (
    _CHEMICAL_CODEC,
    _INDEX_MAP_CODEC,
    _TOPOLOGY_CODEC,
    _TEMPLATE_CODEC,
    _FILE_BYTES_CODEC,
)


@dataclass(frozen=True)
class LegacyVariable:
    identity: str
    segment_name: str
    aliases: tuple[str, ...]
    kind: Literal["protein", "rna", "dna", "chemical"]
    # Independent of Domain.fixed_tokens: a union of local locks can cover
    # every token without making the native segment an is_fixed segment.
    is_fixed: bool


@dataclass(frozen=True)
class LegacyAssembly:
    identity: str
    protomers: tuple[str, ...]
    chains: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class LegacyTranslation:
    """A content-addressed snapshot of the legacy system and its bindings."""

    system: System
    candidate: Candidate
    variables: tuple[LegacyVariable, ...]
    assemblies: tuple[LegacyAssembly, ...]
    state: str


@dataclass(frozen=True)
class _ObjectSnapshot:
    value: Any
    attributes: dict[str, Any]
    references: dict[str, Any]


@dataclass(frozen=True)
class _ManagerSnapshot:
    value: Any
    segment_mapping: dict[str, Any]
    unique_mapping: dict[int, Any]
    primary_mapping: dict[int, str]
    segments: tuple[tuple[str, Any], ...]
    unique_segments: tuple[tuple[int, Any], ...]
    primary_names: tuple[tuple[int, str], ...]
    pre_update: dict[int, tuple[str, list[Any] | None]] | None
    pre_update_reference: Any


@dataclass(frozen=True)
class _ConstantSnapshot:
    name: str
    initialized: bool
    value: Any
    reference: Any


@dataclass(frozen=True)
class _RegistrySnapshot:
    name: str
    value: Any
    entries: tuple[tuple[str, Any], ...] | None
    initialized: bool


@dataclass(frozen=True)
class LegacyStateSnapshot:
    """Exact in-place snapshot of bridge-owned legacy mutable state."""

    segments: tuple[_ObjectSnapshot, ...]
    protomers: tuple[_ObjectSnapshot, ...]
    manager: _ManagerSnapshot
    constants: tuple[_ConstantSnapshot, ...]
    variables: tuple[tuple[str, Any], ...]
    registrations: tuple[_RegistrySnapshot, ...]
    variable_references: tuple[tuple[str, Any], ...]
    graph: _LegacyObjectGraph


@dataclass(frozen=True)
class LegacyAssemblyDependency:
    assembly: str
    occurrences: tuple[Key, ...]
    variables: tuple[tuple[str, Key], ...]


@dataclass(frozen=True)
class LegacyEvidenceDependencies:
    evidence: Evidence
    candidate: Candidate
    system: System
    assemblies: tuple[LegacyAssemblyDependency, ...]


@dataclass(frozen=True)
class LegacyAssemblyValidity:
    assembly: str
    status: Literal["current", "reference", "invalid"]
    changed_variables: tuple[str, ...]
    rebind_validated: bool


def _same_scientific_value(left: Any, right: Any, seen=None) -> bool:
    """Compare snapshots without array truth coercion or NaN inequality."""
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    seen = set() if seen is None else seen
    pair = (id(left), id(right))
    if pair in seen:
        return True
    seen.add(pair)
    import numpy as np

    if isinstance(left, np.ndarray):
        try:
            equal = bool(np.array_equal(left, right, equal_nan=True))
        except TypeError:
            equal = bool(np.array_equal(left, right))
        return (
            left.dtype == right.dtype and left.shape == right.shape
            and left.flags.writeable == right.flags.writeable
            and equal
        )
    if isinstance(left, dict):
        return tuple(left) == tuple(right) and all(
            _same_scientific_value(left[key], right[key], seen) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_scientific_value(a, b, seen) for a, b in zip(left, right)
        )
    if isinstance(left, (float, np.floating)) and math.isnan(left) and math.isnan(right):
        return True
    if not isinstance(left, Enum) and hasattr(left, "__dict__") and hasattr(right, "__dict__"):
        return _same_scientific_value(vars(left), vars(right), seen)
    try:
        if bool(left == right):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _snapshot_value(value: Any) -> Any:
    import numpy as np

    memo = {}
    copied = deepcopy(value, memo)

    def preserve_array_flags(original, seen):
        if id(original) in seen or id(original) not in memo:
            return
        seen.add(id(original))
        if isinstance(original, np.ndarray):
            memo[id(original)].flags.writeable = original.flags.writeable
        elif isinstance(original, dict):
            for item in original.values():
                preserve_array_flags(item, seen)
        elif isinstance(original, (list, tuple, set, frozenset)):
            for item in original:
                preserve_array_flags(item, seen)
        elif hasattr(original, "__dict__"):
            for item in vars(original).values():
                preserve_array_flags(item, seen)

    preserve_array_flags(value, set())
    if not _same_scientific_value(value, copied):
        raise UnsupportedLegacyState(
            f"cannot accurately snapshot {type(value).__name__}; an explicit owner is required"
        )
    return copied


@dataclass(frozen=True)
class _IdentityNode:
    reference: Any
    kind: str
    state: Any


@dataclass(frozen=True)
class _LegacyObjectGraph:
    """A process-local graph of original objects and original outgoing edges.

    Value copies are used only for array contents. Container edges always hold
    their captured original objects, including nodes removed from live roots.
    No current list position or dict value chooses the target of a restore.
    """

    nodes: tuple[_IdentityNode, ...]

    @classmethod
    def capture(cls, roots: tuple[Any, ...]) -> _LegacyObjectGraph:
        import numpy as np
        from astrolabe.core.protomer_config_parser import Protomer, Segment, SegmentManager

        nodes = []
        seen = set()

        def visit(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if type(value) in (type(None), bool, int, float, complex, str, bytes, range) or isinstance(value, (Enum, Path)):
                return
            if type(value) is slice:
                for item in (value.start, value.stop, value.step):
                    visit(item)
                return
            if isinstance(value, np.generic) and not isinstance(value, np.void) and not value.dtype.hasobject:
                return
            if type(value) is np.ndarray:
                if (
                    value.dtype.hasobject or value.dtype.fields is not None or value.dtype.metadata
                    or (value.base is not None and type(value.base) not in (np.ndarray, bytes))
                ):
                    raise UnsupportedLegacyState(
                        "cannot snapshot an array with object fields or unowned backing storage"
                    )
                nodes.append(_IdentityNode(value, "array", (
                    value.copy(order="K"), value.shape, value.strides, value.dtype,
                    value.flags.writeable, value.base, value.ctypes.data,
                )))
                visit(value.base)
                return
            if type(value) is dict:
                entries = tuple(value.items())
                nodes.append(_IdentityNode(value, "dict", entries))
                for key, item in entries:
                    visit(key)
                    visit(item)
                return
            if type(value) in (list, set, tuple, frozenset):
                items = tuple(value)
                if type(value) in (list, set):
                    nodes.append(_IdentityNode(value, type(value).__name__, items))
                for item in items:
                    visit(item)
                return
            frozen_record = (
                not isinstance(value, type) and is_dataclass(value)
                and value.__dataclass_params__.frozen
                and type(value).__module__.startswith("astrolabe.")
            )
            if type(value) in (Protomer, Segment, SegmentManager) or frozen_record:
                if hasattr(value, "__dict__"):
                    attributes = vars(value)
                    nodes.append(_IdentityNode(value, "object", attributes))
                    visit(attributes)
                else:
                    attributes = tuple((field.name, getattr(value, field.name)) for field in fields(value))
                    nodes.append(_IdentityNode(value, "record", attributes))
                    for _, item in attributes:
                        visit(item)
                return
            raise UnsupportedLegacyState(
                f"cannot snapshot object graph node {type(value).__name__}; an explicit owner is required"
            )

        for root in roots:
            visit(root)
        return cls(tuple(nodes))

    def restore(self) -> None:
        import numpy as np

        arrays = [node for node in self.nodes if node.kind == "array"]
        # Diagnose storage replacement before writing any captured graph nodes.
        # Native scientific operations may update contents, not reallocate the
        # storage behind externally shared arrays during a protected session.
        for node in arrays:
            array = node.reference
            _, shape, strides, dtype, _, base, pointer = node.state
            if (
                array.shape != shape or array.strides != strides or array.dtype != dtype
                or array.base is not base or array.ctypes.data != pointer
            ):
                raise UnsupportedLegacyState("array storage/layout changed; cannot restore original object graph")

        def writable(array):
            if type(array.base) is np.ndarray:
                writable(array.base)
            if not array.flags.writeable:
                array.flags.writeable = True

        def depth(node):
            base = node.state[5]
            result = 0
            while type(base) is np.ndarray:
                result += 1
                base = base.base
            return result

        for node in arrays:
            array, saved = node.reference, node.state[0]
            # Numeric equality hides signed zero and NaN payload changes.
            # Equal-sized void views compare/copy each element's exact bytes,
            # preserving content identity even for strided/overlapping views.
            raw = np.dtype((np.void, array.dtype.itemsize))
            equal = bool(np.array_equal(array.view(raw), saved.view(raw)))
            if not equal:
                writable(array)
                np.copyto(array.view(raw), saved.view(raw), casting="no")
        # Restore view flags before their bases, including a writable view of a
        # base that was made read-only after the view was originally created.
        for node in sorted(arrays, key=depth, reverse=True):
            if node.state[4] and not node.reference.flags.writeable:
                writable(node.reference)
            node.reference.flags.writeable = node.state[4]

        attribute_maps = set()
        for node in self.nodes:
            if node.kind == "object":
                attribute_maps.add(id(node.state))
                if vars(node.reference) is not node.state:
                    object.__setattr__(node.reference, "__dict__", node.state)
            elif node.kind == "record":
                for name, reference in node.state:
                    if getattr(node.reference, name) is not reference:
                        object.__setattr__(node.reference, name, reference)
        # Restore frozen record attributes before reinserting hashable record
        # objects into any dictionary or set that uses them as keys.
        for node in self.nodes:
            if node.kind == "dict" and id(node.reference) in attribute_maps:
                node.reference.clear()
                node.reference.update(node.state)
        for node in self.nodes:
            if node.kind == "dict" and id(node.reference) not in attribute_maps:
                node.reference.clear()
                node.reference.update(node.state)
            elif node.kind == "list":
                node.reference[:] = node.state
            elif node.kind == "set":
                node.reference.clear()
                node.reference.update(node.state)


def _require_initialized(name: str, default: Any = None) -> Any:
    try:
        return global_config.get_const(name)
    except RuntimeError:
        return default


def _protomerize_assemblies(
    assemblies: Sequence[str] | None,
    protomers: Mapping[str, Any],
) -> tuple[str, ...]:
    if assemblies is None:
        configured = _require_initialized("num_tokens", None)
        if isinstance(configured, Mapping) and configured:
            assemblies = tuple(str(name) for name in configured)
        else:
            assemblies = tuple(str(name) for name in protomers)
    normalized = tuple(str(name) for name in tuple(assemblies or ()))
    if not normalized:
        raise ValueError("at least one legacy assembly/oligomer is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("legacy assembly names must be unique")
    for name in normalized:
        missing = [protomer for protomer in name if protomer not in protomers]
        if missing:
            raise ValueError(
                f"assembly {name!r} references unknown legacy protomers {missing!r}"
            )
    return normalized


def _canonical_metadata(
    value: Any,
    *,
    path: str = "$",
    data: DataStore | None = None,
) -> Any:
    """Convert supported configuration metadata to deterministic JSON data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise UnsupportedLegacyState(f"non-finite metadata at {path}")
        return value
    if isinstance(value, Path):
        return {"$path": value.as_posix()}
    if isinstance(value, bytes):
        import base64

        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if type(value) is Template:
        if data is None:
            raise UnsupportedLegacyState(
                f"legacy Template at {path} requires a DataStore"
            )
        payload = _TEMPLATE_CODEC.canonical_payload(value)
        reference = data.put(value, _TEMPLATE_CODEC)
        mmcif = payload["mmcif"]
        mapping = payload["query_to_template_map"]
        return {
            _TEMPLATE_REFERENCE: {
                "blob": _blob_record(reference),
                "fields": payload["fields"],
                "mmcif_sha256": (
                    None if mmcif is None else mmcif["sha256"]
                ),
                "query_to_template_map_sha256": mapping["sha256"],
            }
        }

    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return _canonical_metadata(
                OmegaConf.to_container(value, resolve=True),
                path=path,
                data=data,
            )
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _canonical_metadata(value.item(), path=path, data=data)
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise UnsupportedLegacyState(
                    f"object-dtype metadata array at {path} is unsupported"
                )
            return {
                "$ndarray": {
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "data": value.tolist(),
                }
            }
    except ImportError:
        pass

    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise UnsupportedLegacyState(
                    f"non-string metadata key at {path}: {type(key).__name__}"
                )
            result[key] = _canonical_metadata(
                item,
                path=f"{path}.{key}",
                data=data,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _canonical_metadata(item, path=f"{path}[{index}]", data=data)
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        items = [
            _canonical_metadata(
                item,
                path=f"{path}{{set}}",
                data=data,
            )
            for item in value
        ]
        return sorted(items, key=lambda item: key_for(item).digest)
    enum_value = getattr(value, "value", None)
    enum_name = getattr(type(value), "__name__", "")
    if enum_name and enum_value is not None and isinstance(
        enum_value, (str, int, bool)
    ):
        return {"$enum": f"{type(value).__module__}.{type(value).__qualname__}",
                "value": enum_value}
    raise UnsupportedLegacyState(
        f"unsupported legacy metadata at {path}: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _segment_type(segment: Any) -> tuple[str, Literal["protein", "rna", "dna", "chemical"]]:
    type_name = getattr(getattr(segment, "type", None), "name", None)
    if type_name == "PROTEIN":
        return "p", "protein"
    if type_name == "RNA":
        return "r", "rna"
    if type_name == "DNA":
        return "d", "dna"
    if type_name in {"SMILES", "CCD"}:
        return str(getattr(segment.type, "value", type_name)), "chemical"
    raise UnsupportedLegacyState(
        f"unsupported legacy segment type {getattr(segment, 'type', None)!r}"
    )


def _chemical_payload(segment: Any) -> dict[str, Any]:
    type_name = getattr(getattr(segment, "type", None), "name", None)
    if type_name not in {"SMILES", "CCD"}:
        raise UnsupportedLegacyState(f"segment is not a supported chemical: {type_name!r}")
    if not isinstance(getattr(segment, "sequence", None), str):
        raise UnsupportedLegacyState(
            "legacy chemical metadata must be an explicit SMILES/CCD string; "
            f"got {type(getattr(segment, 'sequence', None)).__name__}"
        )
    if not segment.sequence:
        raise UnsupportedLegacyState("legacy chemical value cannot be empty")
    return {
        "schema": "astrolabe.legacy.chemical/v1",
        "kind": type_name.lower(),
        "value": segment.sequence,
        "is_fixed": bool(getattr(segment, "is_fixed", True)),
        "initialized_from_random": bool(
            getattr(segment, "initialized_from_random", False)
        ),
    }


def _decode_chemical(data: DataStore, graph: Blob) -> dict[str, Any]:
    if graph.codec != _CHEMICAL_CODEC.identity:
        raise UnsupportedLegacyState(
            f"chemical graph codec {graph.codec!r} is not a legacy chemical graph"
        )
    value = data.get(graph, _CHEMICAL_CODEC)
    if not isinstance(value, dict) or value.get("schema") != (
        "astrolabe.legacy.chemical/v1"
    ):
        raise UnsupportedLegacyState("invalid legacy chemical graph payload")
    if value.get("kind") not in {"smiles", "ccd"}:
        raise UnsupportedLegacyState(
            f"unsupported legacy chemical kind {value.get('kind')!r}"
        )
    if not isinstance(value.get("value"), str) or not value["value"]:
        raise UnsupportedLegacyState("legacy chemical value must be a non-empty string")
    return value


def _deduplicated_variables(
    protomers: Mapping[str, Any],
    segments: Mapping[str, Any],
    manager: Any,
) -> tuple[LegacyVariable, ...]:
    first_seen: list[tuple[Any, str]] = []
    seen_ids: set[int] = set()

    def include(name: str) -> None:
        segment = segments.get(name)
        if segment is None:
            raise UnsupportedLegacyState(
                f"configured segment {name!r} is absent from global_config.segments"
            )
        if id(segment) not in seen_ids:
            seen_ids.add(id(segment))
            first_seen.append((segment, name))

    for protomer in protomers.values():
        for component in getattr(protomer, "components", ()):
            include(str(component))
    for name in segments:
        include(str(name))

    variables: list[LegacyVariable] = []
    used_identities: set[str] = set()
    for segment, first_name in first_seen:
        primary = None
        if manager is not None:
            primary = getattr(manager, "primary_names", {}).get(id(segment))
        identity = str(primary or first_name)
        if identity in used_identities:
            raise UnsupportedLegacyState(
                f"two distinct legacy segments claim variable identity {identity!r}"
            )
        used_identities.add(identity)
        aliases = tuple(
            str(name)
            for name, candidate_segment in segments.items()
            if candidate_segment is segment
        )
        _, kind = _segment_type(segment)
        variables.append(
            LegacyVariable(
                identity=identity,
                segment_name=str(first_name),
                aliases=aliases,
                kind=kind,
                is_fixed=bool(getattr(segment, "is_fixed", False)),
            )
        )
    return tuple(variables)


def _locked_tokens_by_segment(
    protomers: Mapping[str, Any],
    segments: Mapping[str, Any],
) -> dict[int, set[int]]:
    """Project all protomer locks onto shared polymer objects, never aliases.

    Offsets follow the actual component list, including repeated occurrences.
    Chemical token/atom extents are not a residue coordinate system; reject
    locks in such a protomer rather than infer offsets from a SMILES string.
    """

    result: dict[int, set[int]] = {}
    for name, protomer in protomers.items():
        raw = getattr(protomer, "_locked_positions", ())
        if not isinstance(raw, Collection) or isinstance(raw, (str, bytes, Mapping)):
            raise UnsupportedLegacyState(
                f"protomer {name!r} locks must be a finite collection of integer positions"
            )
        if any(isinstance(position, bool) or not isinstance(position, Integral) for position in raw):
            raise UnsupportedLegacyState(
                f"protomer {name!r} locks must contain integer positions without coercion"
            )
        positions = {int(position) for position in raw}
        if not positions:
            continue

        offset = 0
        spans = []
        for component in protomer.components:
            segment = segments.get(component)
            if segment is None:
                raise UnsupportedLegacyState(
                    f"protomer {name!r} locks reference missing component {component!r}"
                )
            _, kind = _segment_type(segment)
            if kind == "chemical":
                raise UnsupportedLegacyState(
                    f"protomer {name!r} locks cannot be mapped through chemical "
                    f"component {component!r}; only polymer residue offsets are supported"
                )
            if not isinstance(segment.sequence, str):
                raise UnsupportedLegacyState(
                    f"protomer {name!r} locks require an explicit polymer sequence"
                )
            stop = offset + len(segment.sequence)
            spans.append((segment, offset, stop))
            offset = stop
        if any(position < 0 or position >= offset for position in positions):
            raise UnsupportedLegacyState(
                f"protomer {name!r} lock position is out of range for polymer length {offset}"
            )
        for segment, start, stop in spans:
            local = {position - start for position in positions if start <= position < stop}
            if local:
                result.setdefault(id(segment), set()).update(local)
    return result


@scientific_session
def translate_legacy_state(
    *,
    assemblies: Sequence[str] | None = None,
    state: str = "legacy",
    data: DataStore | None = None,
) -> LegacyTranslation:
    """Translate current shared Segment/Protomer state to System + Candidate."""

    data = data if data is not None else DataStore()
    if not isinstance(state, str) or not state:
        raise ValueError("legacy state label must be a non-empty string")

    protomers = _require_initialized("protomers", None)
    segments = _require_initialized("segments", None)
    manager = _require_initialized("segment_manager", None)
    if not isinstance(protomers, Mapping) or not isinstance(segments, Mapping):
        raise UnsupportedLegacyState(
            "global_config.protomers and global_config.segments must be initialized"
        )
    if manager is None:
        raise UnsupportedLegacyState("global_config.segment_manager must be initialized")

    assembly_names = _protomerize_assemblies(assemblies, protomers)
    variables = _deduplicated_variables(protomers, segments, manager)
    variable_by_segment_id = {
        id(segments[variable.segment_name]): variable for variable in variables
    }
    # Deliberately include protomers absent from the selected assemblies: their
    # locks still constrain any segment shared with an evaluated protomer.
    locked_tokens = _locked_tokens_by_segment(protomers, segments)

    assignments: list[Assignment] = []
    domains: list[Domain] = []
    for variable in variables:
        segment = segments[variable.segment_name]
        if variable.kind == "chemical":
            graph = data.put(_chemical_payload(segment), _CHEMICAL_CODEC)
            assignments.append(Assignment(variable.identity, Chemical(graph)))
            continue

        modifications = tuple(
            sorted((int(index), str(name)) for index, name in (
                getattr(segment, "ptm_dict", None) or {}
            ).items())
        )
        assignment = Assignment(
            variable.identity,
            Polymer(variable.kind, tuple(str(segment.sequence)), modifications),
        )
        assignments.append(assignment)
        fixed = (
            tuple(range(len(assignment.value.tokens)))
            if variable.is_fixed
            else tuple(sorted(locked_tokens.get(id(segment), ())))
        )
        if fixed:
            domains.append(
                Domain(
                    variable=variable.identity,
                    fixed_tokens=fixed,
                    fixed_atoms=(),
                    allowed_values=None,
                )
            )

    assembly_payloads: list[dict[str, Any]] = []
    occurrences: list[Occurrence] = []
    legacy_assemblies: list[LegacyAssembly] = []
    for assembly_name in assembly_names:
        chain_ids = chain_ids_for_oligo(assembly_name)
        chain_entries: list[dict[str, Any]] = []
        assembly_offset = 0
        for chain_id, protomer_name in zip(chain_ids, assembly_name):
            protomer = protomers[protomer_name]
            component_entries: list[dict[str, Any]] = []
            protomer_offset = 0
            for component_index, component_name in enumerate(protomer.components):
                segment = segments[component_name]
                variable = variable_by_segment_id[id(segment)]
                variable_length = (
                    1 if variable.kind == "chemical" else len(segment.sequence)
                )
                index_payload = {
                    "schema": "astrolabe.legacy.index-map/v1",
                    "assembly": assembly_name,
                    "chain": chain_id,
                    "protomer": protomer_name,
                    "component": str(component_name),
                    "component_index": component_index,
                    "variable": variable.identity,
                    "variable_range": [0, variable_length],
                    "protomer_range": [
                        protomer_offset,
                        protomer_offset + variable_length,
                    ],
                    "assembly_range": [
                        assembly_offset,
                        assembly_offset + variable_length,
                    ],
                }
                if getattr(protomer, "type", None).name == "PROTEIN":
                    locked = sorted(
                        int(position)
                        for position in getattr(protomer, "_locked_positions", ())
                        if protomer_offset <= int(position) < protomer_offset + variable_length
                    )
                    index_payload["locked_positions"] = locked
                index_map = data.put(index_payload, _INDEX_MAP_CODEC)
                occurrence_identity = key_for(
                    (
                        "astrolabe.design.legacy.occurrence/v1",
                        assembly_name,
                        chain_id,
                        protomer_name,
                        component_index,
                        variable.identity,
                        index_map,
                    )
                )
                occurrence = Occurrence(
                    identity=occurrence_identity,
                    state=state,
                    assembly=assembly_name,
                    chain=chain_id,
                    component=variable.identity,
                    variable=variable.identity,
                    index_map=index_map,
                )
                occurrences.append(occurrence)
                component_entries.append(
                    {
                        "component": str(component_name),
                        "component_index": component_index,
                        "variable": variable.identity,
                        "variable_length": variable_length,
                        "protomer_offset": protomer_offset,
                        "assembly_offset": assembly_offset,
                        "index_map": _blob_record(index_map),
                    }
                )
                assembly_offset += variable_length
                protomer_offset += variable_length
            chain_entries.append(
                {
                    "chain": chain_id,
                    "protomer": protomer_name,
                    "components": component_entries,
                }
            )
        assembly_payloads.append(
            {"name": assembly_name, "chains": chain_entries, "residue_length": assembly_offset}
        )
        legacy_assemblies.append(
            LegacyAssembly(
                identity=assembly_name,
                protomers=tuple(assembly_name),
                chains=tuple(zip(chain_ids, assembly_name)),
            )
        )

    topology_payload = {
        "schema": "astrolabe.legacy.topology/v1",
        "assemblies": assembly_payloads,
        "protomers": {
            str(name): {
                "type": str(getattr(getattr(protomer, "type", None), "value", "")),
                "components": list(getattr(protomer, "components", ())),
                "locked_positions": sorted(
                    int(position)
                    for position in getattr(protomer, "_locked_positions", ())
                ),
                "mask_mutable_msa": bool(
                    getattr(protomer, "_mask_mutable_msa", True)
                ),
                "homology": _canonical_metadata(
                    getattr(protomer, "_homology", {}),
                    path=f"protomers.{name}.homology",
                    data=data,
                ),
            }
            for name, protomer in protomers.items()
        },
        "covalent_bonds": _canonical_metadata(
            _require_initialized("covalent_bonds", None),
            path="covalent_bonds",
            data=data,
        ),
        "cyclization": _canonical_metadata(
            _require_initialized("cyclization", None),
            path="cyclization",
            data=data,
        ),
        "nonstandard_aliases": _canonical_metadata(
            _require_initialized("nonstandard_aliases", {}),
            path="nonstandard_aliases",
            data=data,
        ),
        "user_ccd": _canonical_metadata(
            _require_initialized("user_ccd", None),
            path="user_ccd",
            data=data,
        ),
        "ligand_resnames": _canonical_metadata(
            _require_initialized("ligand_resnames", None),
            path="ligand_resnames",
            data=data,
        ),
    }
    topology = data.put(topology_payload, _TOPOLOGY_CODEC)
    system = make_system(
        tuple(variable.identity for variable in variables),
        tuple(occurrences),
        topology,
        tuple(domains),
    )
    candidate = make_candidate(system, tuple(assignments), data=data)
    return LegacyTranslation(
        system=system,
        candidate=candidate,
        variables=variables,
        assemblies=tuple(legacy_assemblies),
        state=state,
    )


def capture_legacy_candidate(
    *,
    assemblies: Sequence[str] | None = None,
    state: str = "legacy",
    data: DataStore | None = None,
) -> LegacyTranslation:
    return translate_legacy_state(assemblies=assemblies, state=state, data=data)


class LegacyContext:
    """Serialized owner for legacy mutable globals and candidate application."""

    def __init__(self) -> None:
        self._translations: dict[str, LegacyTranslation] = {}
        self._active = False
        self._owner_thread: int | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def owner_thread(self) -> int | None:
        return self._owner_thread

    def capture_candidate(
        self,
        *,
        assemblies: Sequence[str] | None = None,
        state: str = "legacy",
        data: DataStore | None = None,
    ) -> LegacyTranslation:
        with _LEGACY_OWNER:
            translation = translate_legacy_state(
                assemblies=assemblies,
                state=state,
                data=data,
            )
            self._translations[translation.system.identity.digest] = translation
            return translation

    def _translation_for(
        self,
        candidate: Candidate,
        translation: LegacyTranslation | None,
    ) -> LegacyTranslation:
        if translation is not None:
            if translation.candidate.system != candidate.system:
                raise ValueError("translation and candidate belong to different systems")
            return translation
        found = self._translations.get(candidate.system.digest)
        if found is None:
            raise ValueError(
                "candidate system is unknown to this LegacyContext; capture it first "
                "or pass the originating translation"
            )
        return found

    def apply_candidate(
        self,
        candidate: Candidate,
        *,
        translation: LegacyTranslation | None = None,
        data: DataStore | None = None,
    ) -> None:
        """Apply a complete candidate atomically, restoring on any exception."""

        with _LEGACY_OWNER:
            found = self._translation_for(candidate, translation)
            active_data = data
            if active_data is None:
                active_data = DataStore()
            snapshot = self.capture_state()
            try:
                self._apply_candidate(candidate, found, active_data)
            except BaseException:
                self.restore_state(snapshot)
                raise

    def _apply_candidate(
        self,
        candidate: Candidate,
        translation: LegacyTranslation,
        data: DataStore,
    ) -> None:
        if candidate.system != translation.system.identity:
            raise ValueError("candidate belongs to a different system")
        canonical = make_candidate(translation.system, candidate.assignments, data=data)
        if canonical.identity != candidate.identity:
            raise ValueError("candidate identity does not match its content")
        # Validate edits against the captured base even for callers constructing
        # Candidate records directly, preserving fixed tokens, PTMs and domains.
        translation.system.edit(
            translation.candidate, canonical.assignments, data=data,
        )

        segments = _require_initialized("segments", None)
        protomers = _require_initialized("protomers", None)
        manager = _require_initialized("segment_manager", None)
        if not isinstance(segments, Mapping) or not isinstance(protomers, Mapping):
            raise UnsupportedLegacyState("legacy segment/protomer globals are unavailable")

        variables = {variable.identity: variable for variable in translation.variables}
        new_assignments = {
            assignment.variable: assignment for assignment in candidate.assignments
        }
        domains = {domain.variable: domain for domain in translation.system.domains}
        live_locks = _locked_tokens_by_segment(protomers, segments)
        for variable in translation.variables:
            segment = segments[variable.segment_name]
            domain = domains.get(variable.identity)
            captured = set(() if domain is None else domain.fixed_tokens)
            if not live_locks.get(id(segment), set()).issubset(captured):
                raise UnsupportedLegacyState(
                    f"legacy locks changed for {variable.identity!r}; "
                    "capture a new translation before applying a candidate"
                )
        changed_segments: list[Any] = []

        for variable_id in translation.system.variables:
            new = new_assignments[variable_id]
            variable = variables[variable_id]
            segment = segments[variable.segment_name]
            if any(segments.get(alias) is not segment for alias in variable.aliases):
                raise UnsupportedLegacyState(
                    f"legacy aliases for {variable_id!r} no longer share one segment"
                )
            if variable.kind == "chemical":
                if not isinstance(new.value, Chemical):
                    raise UnsupportedLegacyState(
                        f"chemical variable {variable_id!r} cannot accept polymer state"
                    )
                payload = _decode_chemical(data, new.value.graph)
                expected_kind = getattr(segment.type, "name", "").lower()
                if payload["kind"] != expected_kind:
                    raise UnsupportedLegacyState(
                        f"variable {variable_id!r} expects {expected_kind!r} chemistry, "
                        f"got {payload['kind']!r}"
                    )
                if _chemical_payload(segment) == payload:
                    continue
                segment.sequence = payload["value"]
                segment.ptm_dict = {}
                segment.is_fixed = bool(payload.get("is_fixed", True))
            else:
                if not isinstance(new.value, Polymer):
                    raise UnsupportedLegacyState(
                        f"polymer variable {variable_id!r} cannot accept chemical state"
                    )
                current = Polymer(
                    _segment_type(segment)[1], tuple(segment.sequence),
                    tuple(sorted((segment.ptm_dict or {}).items())),
                )
                if current.kind != variable.kind:
                    raise UnsupportedLegacyState(
                        f"legacy molecule kind changed for {variable_id!r}"
                    )
                if current == new.value and bool(segment.is_fixed) == variable.is_fixed:
                    continue
                segment.sequence = "".join(new.value.tokens)
                segment.ptm_dict = dict(new.value.modifications)
                segment.is_fixed = variable.is_fixed
            segment.probs = None
            segment.scores = None
            segment.mutations = []
            changed_segments.append(segment)

        if not changed_segments:
            return
        manager._pre_update_snapshot = None
        changed_ids = {id(segment) for segment in changed_segments}
        for protomer in protomers.values():
            if not any(id(segments[name]) in changed_ids for name in protomer.components):
                continue
            protomer._ptm_dict = {}
            for attribute in (
                "_mutable_mask",
                "_homology_seq",
                "_tied_positions",
                "_tied_positions_dict",
            ):
                if hasattr(protomer, attribute):
                    delattr(protomer, attribute)

    def capture_state(self) -> LegacyStateSnapshot:
        """Capture all bridge-owned scientific state for exact in-place restore."""

        with _LEGACY_OWNER:
            segments = _require_initialized("segments", {})
            protomers = _require_initialized("protomers", {})
            manager = _require_initialized("segment_manager", None)
            if not isinstance(segments, dict) or not isinstance(protomers, dict):
                raise UnsupportedLegacyState("legacy segment/protomer globals are unavailable")
            if manager is None:
                raise UnsupportedLegacyState("legacy segment manager is unavailable")

            segment_values: list[Any] = []
            seen: set[int] = set()
            for segment in tuple(segments.values()) + tuple(manager.segments.values()) + tuple(
                getattr(manager, "unique_segments", {}).values()
            ):
                if id(segment) not in seen:
                    seen.add(id(segment))
                    segment_values.append(segment)

            # One memoized graph spans every scientific owner and root. This
            # captures aliases across segments, protomers, globals and manager
            # caches, not just aliases within an individual deep-copied value.
            graph = _LegacyObjectGraph.capture((
                segments, protomers, manager,
                _require_initialized("mutable_protomers", None),
                *(global_config.__dict__.get(f"_{name}") for name in sorted(_SNAPSHOT_CONSTANTS)),
                *(global_config.get_variable(name) for name in global_config.variable_names),
            ))

            constants: list[_ConstantSnapshot] = []
            for name in sorted(_SNAPSHOT_CONSTANTS):
                initialized = bool(global_config._initialized_constants.get(name, False))
                value = global_config.__dict__.get(f"_{name}")
                try:
                    copied = _snapshot_value(value)
                except Exception as error:
                    raise UnsupportedLegacyState(
                        f"cannot snapshot legacy constant {name!r}: {error}"
                    ) from error
                constants.append(_ConstantSnapshot(name, initialized, copied, value))

            variables: list[tuple[str, Any]] = []
            variable_references = []
            for name in global_config.variable_names:
                value = global_config.get_variable(name)
                try:
                    copied = _snapshot_value(value)
                except Exception as error:
                    raise UnsupportedLegacyState(
                        f"cannot snapshot legacy variable {name!r}: {error}"
                    ) from error
                variables.append((name, copied))
                variable_references.append((name, value))

            pre_update = getattr(manager, "_pre_update_snapshot", None)
            registrations = []
            for name in ("segments", "protomers", "mutable_protomers"):
                initialized = global_config._initialized_constants[name]
                mapping = _require_initialized(name, None)
                if initialized and not isinstance(mapping, dict):
                    raise UnsupportedLegacyState(f"legacy registry {name!r} must be a dictionary")
                registrations.append(_RegistrySnapshot(
                    name, mapping, tuple(mapping.items()) if initialized else None, initialized,
                ))
            return LegacyStateSnapshot(
                segments=tuple(
                    self._capture_object(segment, "segment") for segment in segment_values
                ),
                protomers=tuple(
                    self._capture_object(protomer, "protomer")
                    for protomer in protomers.values()
                ),
                manager=_ManagerSnapshot(
                    value=manager,
                    segment_mapping=manager.segments,
                    unique_mapping=manager.unique_segments,
                    primary_mapping=manager.primary_names,
                    segments=tuple(manager.segments.items()),
                    unique_segments=tuple(
                        (identity, segment)
                        for identity, segment in getattr(
                            manager, "unique_segments", {}
                        ).items()
                    ),
                    primary_names=tuple(
                        (identity, name)
                        for identity, name in getattr(
                            manager, "primary_names", {}
                        ).items()
                    ),
                    pre_update=_snapshot_value(pre_update),
                    pre_update_reference=pre_update,
                ),
                constants=tuple(constants),
                variables=tuple(variables),
                registrations=tuple(registrations),
                variable_references=tuple(variable_references),
                graph=graph,
            )

    @staticmethod
    def _capture_object(value: Any, label: str) -> _ObjectSnapshot:
        try:
            references = dict(vars(value))
            attributes = _snapshot_value(references)
        except Exception as error:
            raise UnsupportedLegacyState(
                f"cannot snapshot legacy {label} {type(value).__name__}: {error}"
            ) from error
        return _ObjectSnapshot(value=value, attributes=attributes, references=references)

    def restore_state(self, snapshot: LegacyStateSnapshot) -> None:
        """Restore a state captured by :meth:`capture_state`."""

        if not isinstance(snapshot, LegacyStateSnapshot):
            raise TypeError("snapshot must be a LegacyStateSnapshot")
        with _LEGACY_OWNER:
            snapshot.graph.restore()
            for registration in snapshot.registrations:
                global_config.__dict__[f"_{registration.name}"] = registration.value
                global_config._initialized_constants[registration.name] = registration.initialized
            manager = snapshot.manager.value
            global_config.__dict__["_segment_manager"] = manager
            global_config._initialized_constants["segment_manager"] = True

            for constant in snapshot.constants:
                global_config.__dict__[f"_{constant.name}"] = constant.reference
                global_config._initialized_constants[constant.name] = constant.initialized
            variable_references = dict(snapshot.variable_references)
            for name, _ in snapshot.variables:
                global_config.set_variable(name, variable_references[name])

    @contextmanager
    def activate(
        self,
        candidate: Candidate | None = None,
        *,
        translation: LegacyTranslation | None = None,
        data: DataStore | None = None,
    ):
        """Own serialized legacy state; always restore it on context exit."""

        with _LEGACY_OWNER:
            if self._active:
                raise RuntimeError(
                    "LegacyContext does not provide nested or parallel isolation"
                )
            active_data = data if data is not None else DataStore()
            found = (
                self._translation_for(candidate, translation)
                if candidate is not None else translation
            )
            if found is not None:
                _validate_candidate(found.system, found.candidate, data=active_data)
                current = translate_legacy_state(
                    assemblies=tuple(assembly.identity for assembly in found.assemblies),
                    state=found.state,
                    data=active_data,
                )
                if (
                    current.system != found.system
                    or current.variables != found.variables
                    or current.assemblies != found.assemblies
                ):
                    raise UnsupportedLegacyState(
                        "current legacy System differs from the captured System "
                        "(topology, occurrences or locks); recapture before activation"
                    )
            snapshot = self.capture_state()
            self._active = True
            self._owner_thread = threading.get_ident()
            try:
                if candidate is not None:
                    self._apply_candidate(candidate, found, active_data)
                yield self
            finally:
                try:
                    self.restore_state(snapshot)
                finally:
                    self._active = False
                    self._owner_thread = None


def _assignment_stamp(assignment: Assignment) -> Key:
    return key_for(("astrolabe.design.legacy.assignment/v1", assignment))


def _validate_candidate(
    system: System, candidate: Candidate, *, data: DataStore | None = None,
) -> None:
    canonical_system = make_system(
        system.variables, system.occurrences, system.topology, system.domains,
    )
    if canonical_system != system:
        raise ValueError("System identity does not match its content")
    if candidate.system != system.identity:
        raise ValueError("candidate belongs to a different System")
    if make_candidate(system, candidate.assignments, data=data).identity != candidate.identity:
        raise ValueError("candidate identity does not match its content")


def project_assembly_dependencies(
    system: System,
    candidate: Candidate,
    *,
    assemblies: Sequence[str] | None = None,
    data: DataStore | None = None,
) -> tuple[LegacyAssemblyDependency, ...]:
    _validate_candidate(system, candidate, data=data)
    selected = None if assemblies is None else set(str(name) for name in assemblies)
    assignment_map = {
        assignment.variable: _assignment_stamp(assignment)
        for assignment in candidate.assignments
    }
    grouped: dict[str, list[Occurrence]] = {}
    for occurrence in system.occurrences:
        if selected is not None and occurrence.assembly not in selected:
            continue
        grouped.setdefault(occurrence.assembly, []).append(occurrence)
    result = []
    for assembly_name in sorted(grouped):
        occurrences = tuple(grouped[assembly_name])
        variable_names = tuple(sorted({item.variable for item in occurrences}))
        result.append(
            LegacyAssemblyDependency(
                assembly=assembly_name,
                occurrences=tuple(item.identity for item in occurrences),
                variables=tuple(
                    (variable, assignment_map[variable])
                    for variable in variable_names
                ),
            )
        )
    if selected is not None and selected != set(grouped):
        raise ValueError(
            f"unknown legacy assemblies requested: {sorted(selected - set(grouped))!r}"
        )
    return tuple(result)


def project_evidence_dependencies(
    evidence: Evidence,
    system: System,
    candidate: Candidate,
    *,
    assemblies: Sequence[str] | None = None,
    data: DataStore | None = None,
) -> LegacyEvidenceDependencies:
    _validate_evidence_source(evidence, system, candidate)
    if evidence.evaluated_candidate != candidate.identity:
        raise ValueError(
            "Evidence subjects are tied to a different evaluated candidate; "
            "use EvidenceUse.relation='reference' instead of relabeling them"
        )
    return LegacyEvidenceDependencies(
        evidence=evidence,
        candidate=candidate,
        system=system,
        assemblies=project_assembly_dependencies(
            system, candidate, assemblies=assemblies, data=data,
        ),
    )


def _validate_evidence_source(evidence: Evidence, system: System, candidate: Candidate) -> None:
    if evidence.with_measurements(()).identity != evidence.identity:
        raise ValueError("Evidence identity does not match its source content")
    if manifest(evidence.inputs.dependencies) != evidence.inputs:
        raise ValueError("Evidence input identity does not match its source content")
    for dependency in evidence.inputs.dependencies:
        if dependency.role == "system" and dependency.identity != system.identity:
            raise ValueError("Evidence System dependency does not match its source")
        if dependency.role == "candidate" and dependency.identity != candidate.identity:
            raise ValueError("Evidence candidate dependency does not match its source")
    occurrences = {item.identity: item for item in system.occurrences}
    for artifact in evidence.artifacts:
        if artifact.candidate != candidate.identity or any(
            occurrences.get(item.identity) != item for item in artifact.occurrences
        ):
            raise ValueError("Evidence artifact origin does not match its source System/candidate")
    for measurement in evidence.measurements:
        for subject in measurement.subjects:
            if subject.candidate != candidate.identity or any(
                identity not in occurrences
                or (subject.assembly is not None and occurrences[identity].assembly != subject.assembly)
                for identity in subject.occurrences
            ):
                raise ValueError("Evidence measurement origin does not match its source System/candidate")


def invalidate_evidence_projection(
    projection: LegacyEvidenceDependencies,
    system: System,
    edited: Candidate,
    *,
    data: DataStore | None = None,
) -> tuple[LegacyAssemblyValidity, ...]:
    """Invalidate coupled assemblies; leave other assemblies as references."""

    _validate_candidate(projection.system, projection.candidate, data=data)
    _validate_candidate(system, edited, data=data)
    _validate_evidence_source(projection.evidence, projection.system, projection.candidate)
    if projection.system != system:
        raise ValueError("Evidence reuse requires the same complete System, topology and occurrences")
    system.edit(projection.candidate, edited.assignments, data=data)
    if projection.evidence.evaluated_candidate != projection.candidate.identity:
        raise ValueError("Evidence evaluated candidate identity does not match the projection")
    expected = project_assembly_dependencies(
        projection.system, projection.candidate,
        assemblies=tuple(item.assembly for item in projection.assemblies), data=data,
    )
    if expected != projection.assemblies:
        raise ValueError("projected assembly dependencies do not match their source content")
    original_assignments = {item.variable: item for item in projection.candidate.assignments}
    global_changes = tuple(
        item.variable for item in edited.assignments
        if original_assignments[item.variable] != item
    )
    unscoped = any(
        subject.assembly is None
        for measurement in projection.evidence.measurements for subject in measurement.subjects
    )
    current = {
        dependency.assembly: dependency
        for dependency in project_assembly_dependencies(
            system,
            edited,
            assemblies=tuple(
                dependency.assembly for dependency in projection.assemblies
            ),
            data=data,
        )
    }
    result = []
    for previous in projection.assemblies:
        before = dict(previous.variables)
        after = dict(current[previous.assembly].variables)
        changed = tuple(
            variable
            for variable, stamp in before.items()
            if after.get(variable) != stamp
        )
        if unscoped:
            # A legacy aggregate has no proven assembly-local dependency map.
            # Do not silently reuse that record after a different variable edits.
            changed = tuple(sorted(set(changed) | set(global_changes)))
        if projection.candidate.identity == edited.identity:
            status: Literal["current", "reference", "invalid"] = "current"
        elif changed:
            status = "invalid"
        else:
            status = "reference"
        result.append(
            LegacyAssemblyValidity(
                assembly=previous.assembly,
                status=status,
                changed_variables=changed,
                rebind_validated=status in {"current", "reference"},
            )
        )
    return tuple(result)


def validate_evidence_rebind(
    projection: LegacyEvidenceDependencies,
    assembly: str,
    system: System,
    edited: Candidate,
    *,
    data: DataStore | None = None,
) -> LegacyAssemblyValidity:
    """Validate reference reuse after an edit; never relabel raw subjects."""

    for item in invalidate_evidence_projection(projection, system, edited, data=data):
        if item.assembly == assembly:
            if not item.rebind_validated:
                raise ValueError(
                    f"assembly {assembly!r} has changed variable dependencies "
                    f"{item.changed_variables!r}; rerun evaluation"
                )
            return item
    raise ValueError(f"unknown projected assembly {assembly!r}")


def _samples_for(
    samples: Sample | Sequence[Sample] | Mapping[str, Sequence[Sample]] | None,
    assembly: str,
    index: int,
) -> Sample | None:
    if samples is None:
        return None
    if isinstance(samples, Sample):
        values: Sequence[Sample] = (samples,)
    elif isinstance(samples, Mapping):
        values = samples.get(assembly, ())
    else:
        values = samples
    values = tuple(values)
    if index >= len(values):
        return None
    value = values[index]
    if not isinstance(value, Sample):
        raise TypeError("samples must contain Sample values")
    return value


def _store_evidence_value(data: DataStore, value: Any) -> Blob | None:
    if value is None:
        return None
    try:
        import numpy as np

        if isinstance(value, np.ndarray) or (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(item, (int, float, np.number)) for item in value)
        ):
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise UnsupportedLegacyState(
                    "object-dtype score arrays cannot be preserved safely"
                )
            return data.put(array, ArrayCodec())
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, np.ndarray):
            return data.put(value, ArrayCodec())
    except ImportError:
        pass
    try:
        native = _canonical_metadata(value, path="score value")
    except UnsupportedLegacyState:
        raise
    return data.put(native, _JSON)


def _numeric_evidence_array(value: Any) -> Any | None:
    """Return the numeric value needing array storage, preserving scalar dtype."""

    try:
        import numpy as np
    except ImportError:
        return None

    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return array if np.issubdtype(array.dtype, np.number) else None
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.number):
            return np.asarray(value)
        return None
    if type(value) in (int, float):
        return np.asarray(value)
    if (
        isinstance(value, (list, tuple))
    ):
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number):
            return array
    return None


def _measurement_inputs(
    system: System,
    candidate: Candidate,
    *,
    state: str,
    sample: Sample | None,
    artifact: Key | None,
    provenance: str,
) -> Any:
    dependencies = [
        Dependency("system", system.identity),
        Dependency("candidate", candidate.identity),
        Dependency("producer", key_for(("legacy.standard_scores/v1", provenance))),
    ]
    if sample is not None:
        dependencies.append(Dependency("sample", key_for(sample)))
    if artifact is not None:
        dependencies.append(Dependency("artifact", artifact))
    if provenance:
        dependencies.append(Dependency("legacy-provenance", key_for(provenance)))
    dependencies.append(Dependency("state", key_for(state)))
    return manifest(tuple(dependencies))


def _make_measurement(
    data: DataStore,
    *,
    system: System,
    candidate: Candidate,
    state: str,
    quantity: str,
    units: str,
    value: Any,
    subject: Subject,
    axes: tuple[Any, ...],
    provenance: str,
    missing: Literal["present", "partial", "unavailable", "failed"] = "present",
    value_codec: Any | None = None,
) -> Measurement:
    inputs = _measurement_inputs(
        system,
        candidate,
        state=state,
        sample=subject.sample,
        artifact=subject.artifact,
        provenance=provenance,
    )
    record = data.put({
        "schema": "astrolabe.legacy.measurement-provenance/v1",
        "description": provenance,
    }, _JSON)
    dependencies = list(inputs.dependencies)
    dependencies.append(Dependency("legacy-provenance-record", record.identity))
    if data.root is not None:
        data.persist((record,))
    definition = key_for(
        (
            "astrolabe.legacy.measurement-definition/v1",
            quantity,
            units,
            provenance,
        )
    )
    producer = key_for(("astrolabe.legacy.standard-scores-producer/v1",))
    valid_mask = None
    numeric = _numeric_evidence_array(value) if value is not None and value_codec is None else None
    if numeric is None:
        # Non-numeric legacy records are not numeric measurements. Preserve the
        # actual source bytes/metadata explicitly, without inventing samples or
        # numbers; the generic numerical reader sees an unavailable value.
        source = (
            data.put(value, value_codec) if value is not None and value_codec is not None
            else _store_evidence_value(data, value)
        )
        if source is not None:
            dependencies.append(Dependency(f"legacy-source-record/{source.codec}", source.identity))
            if data.root is not None:
                data.persist((source,))
        values = None
        axes = ()
    else:
        import numpy as np

        if np.iscomplexobj(numeric):
            raise UnsupportedLegacyState("complex-valued scores are not real numeric measurements")
        values = data.put(numeric, ArrayCodec())
        valid = np.asarray(np.isfinite(numeric), dtype=bool)
        if not numeric.size or not bool(np.all(valid)):
            valid_mask = data.put(valid, ArrayCodec())
            if missing != "failed":
                missing = "partial"
    inputs = manifest(tuple(dependencies))
    missing = missing if values is not None or missing == "failed" else "unavailable"
    return Measurement(
        identity=key_for(
            (
                "measurement/v1",
                quantity,
                definition,
                units,
                producer,
                (subject,),
                tuple(axes),
                values,
                missing,
                valid_mask,
                inputs,
            )
        ),
        quantity=quantity,
        definition=definition,
        units=units,
        producer=producer,
        subjects=(subject,),
        axes=tuple(axes),
        values=values,
        missing=missing,
        valid_mask=valid_mask,
        inputs=inputs,
    )


def _legacy_axes(data: DataStore, value: Any, *, label: str) -> tuple[Any, ...]:
    try:
        import numpy as np

        array = np.asarray(value)
        shape = tuple(int(size) for size in array.shape)
        if not shape:
            return ()
    except Exception:
        return ()
    from .types import Axis

    return tuple(
        Axis(name=f"legacy_axis_{index}", kind="legacy_position", index_map=data.put({
            "schema": "astrolabe.legacy.axis-map/v1",
            "label": label,
            "shape": list(shape),
            "dimension": index,
            "labels": [str(position) for position in range(size)],
            "meaning": "legacy positional indices; residue/atom identity unavailable",
        }, _JSON))
        for index, size in enumerate(shape)
    )


def _verified_oligo_axes(
    data: DataStore,
    system: System,
    candidate: Candidate,
    assembly: str,
    value: Any,
    *,
    label: str,
) -> tuple[Any, ...] | None:
    """Build residue axes only from a complete, verified ordered occurrence map."""

    try:
        import numpy as np

        shape = tuple(int(size) for size in np.asarray(value).shape)
    except Exception:
        return None
    if len(shape) not in (1, 2):
        return None

    assignments = {
        assignment.variable: assignment.value
        for assignment in candidate.assignments
    }
    ordered: list[tuple[int, int, Occurrence, dict[str, Any]]] = []
    for occurrence in system.occurrences:
        if occurrence.assembly != assembly:
            continue
        if occurrence.index_map.codec != _INDEX_MAP_CODEC.identity:
            return None
        payload = data.get(occurrence.index_map, _INDEX_MAP_CODEC)
        if not isinstance(payload, dict):
            return None
        value_range = payload.get("assembly_range")
        if (
            not isinstance(value_range, list)
            or len(value_range) != 2
            or any(type(item) is not int for item in value_range)
        ):
            return None
        ordered.append((value_range[0], value_range[1], occurrence, payload))
    ordered.sort(key=lambda item: (item[0], item[1]))

    entries: list[dict[str, Any]] = []
    expected = 0
    for start, stop, occurrence, payload in ordered:
        assignment = assignments.get(occurrence.variable)
        if (
            start != expected
            or stop <= start
            or not isinstance(assignment, Polymer)
            or stop - start != len(assignment.tokens)
        ):
            return None
        modifications = dict(assignment.modifications)
        for position, token in enumerate(assignment.tokens):
            entry = {
                "assembly_index": start + position,
                "occurrence": occurrence.identity.digest,
                "chain": occurrence.chain,
                "variable": occurrence.variable,
                "variable_index": position,
                "token": token,
            }
            if position in modifications:
                entry["modification"] = modifications[position]
            entries.append(entry)
        expected = stop
    if not entries or any(dimension != len(entries) for dimension in shape):
        return None

    from .types import Axis

    index_map = data.put(
        {
            "schema": "astrolabe.legacy.ordered-residue-map/v1",
            "label": label,
            "shape": list(shape),
            "meaning": "verified ordered residue token and occurrence map",
            "labels": [f"{entry['occurrence']}:{entry['variable_index']}" for entry in entries],
            "entries": entries,
        },
        _JSON,
    )
    if len(shape) == 1:
        return (Axis(name="residue", kind="residue", index_map=index_map),)
    names = ("left_residue", "right_residue")
    return tuple(
        Axis(name=names[index], kind="residue", index_map=index_map)
        for index in range(len(shape))
    )


def _score_axes(
    data: DataStore,
    system: System,
    candidate: Candidate,
    assembly: str,
    value: Any,
    *,
    field: str,
) -> tuple[Any, ...]:
    if field in {"plddt", "pae", "pde"}:
        verified = _verified_oligo_axes(
            data,
            system,
            candidate,
            assembly,
            value,
            label=f"{field}:{assembly}",
        )
        if verified is not None:
            return verified
    return _legacy_axes(data, value, label=f"{field}:{assembly}")


@scientific_session
def standard_scores_to_evidence(
    scores: Any,
    system: System,
    candidate: Candidate,
    data: DataStore,
    *,
    samples: Sample | Sequence[Sample] | Mapping[str, Sequence[Sample]] | None = None,
    artifact_sample_by_assembly: Mapping[str, Sample] | None = None,
    state: str = "legacy",
    producer_label: str = "legacy.StandardScores",
    process_resource: Blob | None = None,
) -> Evidence:
    """Translate StandardScores without inventing copy or sample identity."""

    if not isinstance(data, DataStore):
        raise TypeError("data must be a design DataStore")
    if candidate.system != system.identity:
        raise ValueError("candidate belongs to a different system")
    if make_candidate(system, candidate.assignments, data=data).identity != candidate.identity:
        raise ValueError("candidate identity does not match its content")
    if not isinstance(state, str) or not state:
        raise ValueError("state must be a non-empty string")

    from astrolabe.evaluators.utils import StandardScores
    from astrolabe.score_contributions import normalize_score_contributions

    if not isinstance(scores, StandardScores):
        raise TypeError("scores must be a StandardScores instance")
    if process_resource is not None:
        raise UnsupportedLegacyState(
            "process resource owner is unavailable in this bridge; "
            "live handle support requires the separately qualified owner integration"
        )
    if scores.complex_poses not in (None, {}):
        raise UnsupportedLegacyState(
            "StandardScores.complex_poses contains live Pose objects without a "
            "qualified process resource owner; cross-process Pose recovery is unsupported"
        )

    occurrences_by_assembly: dict[str, tuple[Occurrence, ...]] = {}
    for occurrence in system.occurrences:
        occurrences_by_assembly.setdefault(occurrence.assembly, [])
        occurrences_by_assembly[occurrence.assembly] += (occurrence,)

    names = tuple(str(name) for name in scores.names)
    unknown = [name for name in names if name not in occurrences_by_assembly]
    if unknown:
        raise UnsupportedLegacyState(
            f"score assemblies are absent from the translated System: {unknown!r}"
        )

    artifact_by_assembly: dict[str, Artifact] = {}
    artifacts: list[Artifact] = []
    selected_structure_records: dict[str, tuple[Blob, Blob, Path]] = {}
    artifact_bindings = dict(artifact_sample_by_assembly or {})
    if any(not isinstance(name, str) or not name for name in artifact_bindings):
        raise TypeError("artifact_sample_by_assembly keys must be assembly names")
    if any(not isinstance(value, Sample) for value in artifact_bindings.values()):
        raise TypeError("artifact_sample_by_assembly values must be Sample values")
    unknown_bindings = set(artifact_bindings) - set(scores.complex_paths or {})
    if unknown_bindings:
        raise ValueError(
            "artifact sample bindings reference assemblies without selected "
            f"structures: {sorted(unknown_bindings)!r}"
        )
    for assembly_name, raw_path in (scores.complex_paths or {}).items():
        assembly_name = str(assembly_name)
        if assembly_name not in occurrences_by_assembly:
            raise UnsupportedLegacyState(
                f"complex path belongs to unknown assembly {assembly_name!r}"
            )
        path = Path(raw_path)
        if not path.is_file():
            raise UnsupportedLegacyState(
                f"complex path for assembly {assembly_name!r} is unavailable: {path}"
            )
        content = data.put(path.read_bytes(), _FILE_BYTES_CODEC)
        path_provenance = data.put(
            {
                "schema": "astrolabe.legacy.complex-path-provenance/v1",
                "assembly": assembly_name,
                "path": path.as_posix(),
                "selection_provenance": "legacy RankingSelectedStructure",
                "producing_sample": (
                    None
                    if assembly_name not in artifact_bindings
                    else {
                        "protocol": artifact_bindings[assembly_name].protocol.digest,
                        "replicate": artifact_bindings[assembly_name].replicate,
                        "seed": artifact_bindings[assembly_name].seed,
                        "index": artifact_bindings[assembly_name].index,
                    }
                ),
            },
            _JSON,
        )
        if data.root is not None:
            data.persist((content, path_provenance))
        selected_structure_records[assembly_name] = (
            content,
            path_provenance,
            path,
        )
        sample = artifact_bindings.get(assembly_name)
        if sample is None:
            # A ranking-selected file may be one of many diffusion predictions.
            # Without an explicit producing-sample binding, an Artifact would
            # falsely attach that structure to a request sample.
            continue
        inputs = manifest(
            (
                Dependency("system", system.identity),
                Dependency("candidate", candidate.identity),
                Dependency("sample", key_for(sample)),
                Dependency("content", content.identity),
                Dependency("legacy-artifact-path", path_provenance.identity),
            )
        )
        artifact = Artifact(
            identity=key_for(
                ("artifact/v1", candidate.identity, occurrences_by_assembly[assembly_name],
                 state, sample, content, inputs)
            ),
            candidate=candidate.identity,
            occurrences=occurrences_by_assembly[assembly_name],
            sample=sample,
            content=content,
            derived_from=(),
            inputs=inputs,
        )
        artifact_by_assembly[assembly_name] = artifact
        artifacts.append(artifact)

    measurements: list[Measurement] = []

    def oligo_subject(
        assembly: str,
        sample: Sample | None,
    ) -> Subject:
        artifact = artifact_by_assembly.get(assembly)
        artifact_identity = (
            artifact.identity
            if artifact is not None
            and sample is not None
            and artifact.sample == sample
            else None
        )
        return Subject(
            candidate=candidate.identity,
            state=state,
            assembly=assembly,
            occurrences=tuple(
                occurrence.identity for occurrence in occurrences_by_assembly[assembly]
            ),
            artifact=artifact_identity,
            sample=sample,
        )

    def aggregate_subject(entity: str) -> Subject:
        return Subject(
            candidate=candidate.identity,
            state=state,
            assembly=None,
            occurrences=(),
            artifact=None,
            sample=None,
        )

    def assembly_subject_without_sample(assembly: str) -> Subject:
        return Subject(
            candidate=candidate.identity,
            state=state,
            assembly=assembly,
            occurrences=tuple(
                occurrence.identity for occurrence in occurrences_by_assembly[assembly]
            ),
            artifact=None,
            sample=None,
        )

    def add_oligo_field(field: str, units: str) -> None:
        mapping = getattr(scores, field, None)
        if not mapping:
            return
        for raw_assembly, values in mapping.items():
            assembly = str(raw_assembly)
            if assembly not in occurrences_by_assembly:
                raise UnsupportedLegacyState(
                    f"{field} references unknown assembly {assembly!r}"
                )
            for index, value in enumerate(tuple(values)):
                sample = _samples_for(samples, assembly, index)
                measurements.append(
                    _make_measurement(
                        data,
                        system=system,
                        candidate=candidate,
                        state=state,
                        quantity=(
                            f"{producer_label}.{field}.{assembly}"
                            f".prediction_{index}"
                        ),
                        units=units,
                        value=value,
                        subject=oligo_subject(assembly, sample),
                        axes=_score_axes(
                            data,
                            system,
                            candidate,
                            assembly,
                            value,
                            field=field,
                        ),
                        provenance=(
                            f"StandardScores.{field}[{assembly}]"
                            f" prediction_index={index}"
                        ),
                    )
                )

    add_oligo_field("plddt", "unitless")
    add_oligo_field("ptm", "unitless")
    add_oligo_field("pae", "angstrom")
    add_oligo_field("pde", "angstrom")
    add_oligo_field("iptm", "unitless")

    for assembly_name, metric_map in (scores.chain_pair_scores or {}).items():
        assembly_name = str(assembly_name)
        if assembly_name not in occurrences_by_assembly:
            raise UnsupportedLegacyState(
                f"chain-pair scores reference unknown assembly {assembly_name!r}"
            )
        for metric_name, values in metric_map.items():
            for index, value in enumerate(tuple(values)):
                sample = _samples_for(samples, assembly_name, index)
                measurements.append(
                    _make_measurement(
                        data,
                        system=system,
                        candidate=candidate,
                        state=state,
                        quantity=(
                            f"{producer_label}.chain_pair_scores."
                            f"{assembly_name}.{metric_name}.prediction_{index}"
                        ),
                        units="unitless",
                        value=value,
                        subject=oligo_subject(assembly_name, sample),
                        axes=_legacy_axes(
                            data, value, label=f"chain_pair:{assembly_name}:{metric_name}"
                        ),
                        provenance=(
                            f"StandardScores.chain_pair_scores"
                            f"[{assembly_name}][{metric_name}]"
                            f" prediction_index={index}"
                        ),
                    )
                )

    for field in ("per_proto_scores", "per_proto_score_terms"):
        mapping = getattr(scores, field, None) or {}
        for protomer, values in mapping.items():
            if isinstance(values, Mapping):
                metric_values = values.items()
            else:
                metric_values = (("score", values),)
            for metric_name, records in metric_values:
                for index, value in enumerate(tuple(records)):
                    subject = aggregate_subject(str(protomer))
                    measurements.append(
                        _make_measurement(
                            data,
                            system=system,
                            candidate=candidate,
                            state=state,
                            quantity=(
                                f"legacy.aggregate.{field}.{protomer}.{metric_name}"
                                f".prediction_{index}"
                            ),
                            units="unitless",
                            value=value,
                            subject=subject,
                            axes=_legacy_axes(
                                data,
                                value,
                                label=f"aggregate:{field}:{protomer}:{metric_name}",
                            ),
                            provenance=(
                                "legacy per-protomer aggregate; physical occurrence "
                                "copy and originating assembly are unavailable"
                                f"; prediction_index={index}"
                            ),
                        )
                    )

    for contribution in normalize_score_contributions(scores.score_contributions):
        if not contribution.enabled:
            continue
        if contribution.scope == "oligo":
            assembly = str(contribution.entity)
            if assembly not in occurrences_by_assembly:
                raise UnsupportedLegacyState(
                    f"score contribution references unknown assembly {assembly!r}"
                )
            # ScoreContribution itself carries no sample/artifact slot.  Do not
            # infer that it belongs to the first score array by list position.
            subject = assembly_subject_without_sample(assembly)
            provenance = (
                f"ScoreContribution[{contribution.source}]: oligo {assembly}; "
                "sample and artifact unavailable"
            )
        elif contribution.scope == "protomer":
            subject = aggregate_subject(str(contribution.entity))
            provenance = (
                f"ScoreContribution[{contribution.source}]: legacy protomer "
                f"{contribution.entity}; occurrence mapping unavailable"
            )
        else:
            subject = aggregate_subject("global")
            provenance = f"ScoreContribution[{contribution.source}]: global"
        provenance = _json_bytes({
            "schema": "astrolabe.legacy.score-contribution/v1",
            "description": provenance,
            "source": contribution.source,
            "metric_name": contribution.metric_name,
            "scope": contribution.scope,
            "entity": contribution.entity,
            "shape": contribution.shape,
            "aggregation": contribution.aggregation,
            "flatten_as": contribution.flatten_as,
            "provenance": contribution.provenance,
            "enabled": contribution.enabled,
        }).decode("utf-8")
        measurements.append(
            _make_measurement(
                data,
                system=system,
                candidate=candidate,
                state=state,
                quantity=(
                    f"legacy.contribution.{contribution.source}."
                    f"{contribution.metric_name}"
                ),
                units="unitless",
                value=contribution.value,
                subject=subject,
                axes=_legacy_axes(
                    data,
                    contribution.value,
                    label=f"contribution:{contribution.metric_name}",
                ),
                provenance=provenance,
            )
        )

    for assembly, (content, path_provenance, path) in selected_structure_records.items():
        if assembly in artifact_by_assembly:
            continue
        measurements.append(
            _make_measurement(
                data,
                system=system,
                candidate=candidate,
                state=state,
                quantity=f"legacy.selected_structure_bytes.{assembly}",
                units="bytes",
                value=data.get(content, _FILE_BYTES_CODEC),
                value_codec=_FILE_BYTES_CODEC,
                subject=assembly_subject_without_sample(assembly),
                axes=(),
                provenance=(
                    "StandardScores.complex_paths ranking selection; producing "
                    "sample, prediction index, and selected-artifact mapping are "
                    f"unavailable; path={path}; provenance={path_provenance.identity.digest}"
                ),
            )
        )

    dependencies = (
            Dependency("system", system.identity),
            Dependency("candidate", candidate.identity),
            Dependency(
                "producer",
                key_for(("legacy.StandardScores", producer_label)),
            ),
            Dependency("legacy-protocol", _PROTOCOL),
        )
    evidence_inputs = manifest(dependencies)
    evidence = Evidence(
        identity=key_for(("evidence-pending/v1",)),
        evaluated_candidate=candidate.identity,
        artifacts=tuple(artifacts),
        measurements=(),
        inputs=evidence_inputs,
    )
    return evidence.with_measurements(tuple(measurements))


def candidate_for_mutation(
    translation: LegacyTranslation,
    replacements: Iterable[Assignment],
    *,
    data: DataStore,
) -> Candidate:
    """Build a proposed candidate without mutating globals."""

    return translation.system.edit(
        translation.candidate,
        tuple(replacements),
        data=data,
    )


@scientific_session
def restore_native_sequence(sequence: str, protomer, segments) -> None:
    """Restore the unique shared sequence; repeated segment references update once."""
    offset = 0
    restored = set()
    for name in protomer.components:
        segment = segments[name]
        if id(segment) in restored:
            continue
        restored.add(id(segment))
        length = len(segment.sequence)
        segment.sequence = sequence[offset:offset + length]
        segment.clear_mutations()
        segment.clean_up()
        offset += length


def native_sequence_key(protomer, segments, *, mutated: bool = False) -> str:
    seen = set()
    parts = []
    for name in protomer.components:
        segment = segments[name]
        if id(segment) not in seen:
            parts.append(segment.mutated_sequence if mutated else segment.sequence)
            seen.add(id(segment))
    return "".join(parts)


@scientific_session
def finalize_pending_mutations(manager=None) -> None:
    """Commit queued legacy segment mutations before candidate capture."""

    if manager is None:
        from astrolabe.core import global_config

        manager = global_config.segment_manager
    apply = getattr(manager, "apply_mutations", None)
    if not callable(apply):
        raise RuntimeError("legacy segment manager cannot apply pending mutations")
    apply()



__all__ = [
    "restore_native_sequence",
    "native_sequence_key",
    "finalize_pending_mutations",
    "LEGACY_REFERENCE_CODECS",
    "LegacyAssembly",
    "LegacyAssemblyDependency",
    "LegacyAssemblyValidity",
    "LegacyContext",
    "LegacyEvidenceDependencies",
    "LegacyFileBytesCodec",
    "LegacyStateSnapshot",
    "LegacyTemplateCodec",
    "LegacyTranslation",
    "LegacyVariable",
    "UnsupportedLegacyState",
    "candidate_for_mutation",
    "capture_legacy_candidate",
    "decode_legacy_templates",
    "invalidate_evidence_projection",
    "project_assembly_dependencies",
    "project_evidence_dependencies",
    "standard_scores_to_evidence",
    "translate_legacy_state",
    "validate_evidence_rebind",
]
