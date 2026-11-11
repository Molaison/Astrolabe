"""Content-addressed storage for unified design data.

DataStore owns hashing and persistence. Codecs own canonical byte formats and
the typed references contained in their values.
"""

from __future__ import annotations

import base64
from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Generic, TypeVar
from urllib.parse import quote

from astrolabe.profiling import event, profiled

from . import types as design_types
from .types import Blob, Codec, Key


T = TypeVar("T")


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


def _validate_json_native(value: Any, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON float at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_native(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"JSON object keys at {path} must be strings")
        for key, item in value.items():
            _validate_json_native(item, path=f"{path}[{key!r}]")
        return
    raise TypeError(
        f"value at {path} is not JSON-native: {type(value).__module__}."
        f"{type(value).__qualname__}"
    )


class JSONCodec:
    """Canonical JSON codec for JSON-native dictionaries and collections."""

    def __init__(self, identity: str = "json/v1") -> None:
        if not isinstance(identity, str) or not identity:
            raise ValueError("codec identity must be a non-empty string")
        self.identity = identity

    def encode(self, value: Any) -> bytes:
        _validate_json_native(value, path="$")
        return _json_bytes(value)

    def decode(self, data: bytes) -> Any:
        if not isinstance(data, bytes):
            raise TypeError("encoded JSON data must be bytes")
        return _decode_json(data)

    def references(self, value: Any) -> tuple[Blob, ...]:
        _validate_json_native(value, path="$")
        return ()

    def compatibility_key(self) -> tuple[str, str]:
        return (type(self).__name__, self.identity)


class ArrayCodec:
    """NumPy ``.npy`` codec; decoded arrays are independent and read-only."""

    def __init__(self, identity: str = "numpy/v1") -> None:
        if not isinstance(identity, str) or not identity:
            raise ValueError("codec identity must be a non-empty string")
        self.identity = identity

    def encode(self, value: Any) -> bytes:
        import numpy as np

        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("object dtype arrays are not supported")
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return buffer.getvalue()

    def decode(self, data: bytes) -> Any:
        import numpy as np

        if not isinstance(data, bytes):
            raise TypeError("encoded array data must be bytes")
        array = np.load(io.BytesIO(data), allow_pickle=False)
        array = np.array(array, copy=True, order="C")
        array.setflags(write=False)
        return array

    def references(self, value: Any) -> tuple[Blob, ...]:
        import numpy as np

        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("object dtype arrays are not supported")
        return ()

    def compatibility_key(self) -> tuple[str, str]:
        return (type(self).__name__, self.identity)


class DataclassCodec(Generic[T]):
    """JSON-safe recursive dataclass codec with an explicit type allowlist."""

    def __init__(
        self,
        value_type: type[T] | object,
        identity: str,
        *,
        nested_types: tuple[type[Any], ...] = (),
    ) -> None:
        if not isinstance(identity, str) or not identity:
            raise ValueError("codec identity must be a non-empty string")
        if value_type is not object and not (
            isinstance(value_type, type) and is_dataclass(value_type)
        ):
            raise TypeError("value_type must be a dataclass type or object")

        self.value_type = value_type
        self.identity = identity
        self._sealed = False
        registered: dict[str, type[Any]] = {}
        for dataclass_type in _design_dataclass_types():
            registered[_type_tag(dataclass_type)] = dataclass_type
        if value_type is not object:
            registered[_type_tag(value_type)] = value_type
        for nested_type in nested_types:
            if not isinstance(nested_type, type) or not is_dataclass(nested_type):
                raise TypeError("nested_types entries must be dataclass types")
            registered[_type_tag(nested_type)] = nested_type
        self._types = registered

    def register_types(self, *value_types: type[Any]) -> None:
        """Register additional dataclass types before first encode/decode use."""

        if self._sealed:
            raise RuntimeError("cannot register types after codec use")
        additions: dict[str, type[Any]] = {}
        for value_type in value_types:
            if not isinstance(value_type, type) or not is_dataclass(value_type):
                raise TypeError("registered types must be dataclasses")
            additions[_type_tag(value_type)] = value_type
        self._types.update(additions)

    @property
    def _used(self) -> bool:
        return self._sealed

    def encode(self, value: T) -> bytes:
        self._sealed = True
        # Object roots are validated by the encoding traversal itself. Decode
        # still needs its separate check after dataclass constructors have run.
        if self.value_type is not object:
            self._validate_root(value)
        return _json_bytes(self._encode_value(value, path="$"))

    def decode(self, data: bytes) -> T:
        self._sealed = True
        if not isinstance(data, bytes):
            raise TypeError("encoded dataclass data must be bytes")
        value = self._decode_value(_decode_json(data), path="$")
        self._validate_root(value)
        return value

    def references(self, value: T) -> tuple[Blob, ...]:
        self._sealed = True
        if self.value_type is not object:
            self._validate_root(value)
        return tuple(self._collect_references(value, path="$"))

    def compatibility_key(self) -> tuple[Any, ...]:
        root = "*" if self.value_type is object else _type_tag(self.value_type)
        return (
            type(self).__name__,
            self.identity,
            root,
            tuple(sorted((_type_tag(value_type) for value_type in self._types.values()))),
        )

    def _validate_root(self, value: Any) -> None:
        if self.value_type is object:
            self._validate_structured_value(value, path="$")
            return
        if not is_dataclass(value) or isinstance(value, type):
            raise TypeError(
                f"DataclassCodec expects a {self.value_type.__name__} dataclass"
            )
        if type(value) is not self.value_type:
            raise TypeError(
                f"codec expects {self.value_type.__name__}, got "
                f"{type(value).__name__}"
            )
        tag = _type_tag(type(value))
        if tag not in self._types:
            raise TypeError(f"dataclass type {tag!r} is not registered")

    def _validate_structured_value(self, value: Any, *, path: str) -> None:
        if value is None or isinstance(value, (str, bool, int, bytes, Path)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"non-finite float at {path}")
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            for index, item in enumerate(value):
                self._validate_structured_value(item, path=f"{path}[{index}]")
            return
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError(f"dictionary keys at {path} must be strings")
            for key, item in value.items():
                self._validate_structured_value(item, path=f"{path}[{key!r}]")
            return
        if is_dataclass(value) and not isinstance(value, type):
            tag = _type_tag(type(value))
            if tag not in self._types:
                raise TypeError(f"dataclass type {tag!r} is not registered")
            for field in fields(value):
                self._validate_structured_value(
                    getattr(value, field.name),
                    path=f"{path}.{field.name}",
                )
            return
        raise TypeError(
            f"unsupported structured value at {path}: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )

    def _encode_value(self, value: Any, *, path: str) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"non-finite float at {path}")
            return value
        if isinstance(value, bytes):
            return {"$bytes": base64.b64encode(value).decode("ascii")}
        if isinstance(value, Path):
            return {"$path": value.as_posix()}
        if isinstance(value, tuple):
            return {
                "$tuple": [
                    self._encode_value(item, path=f"{path}[{index}]")
                    for index, item in enumerate(value)
                ]
            }
        if isinstance(value, list):
            return {
                "$list": [
                    self._encode_value(item, path=f"{path}[{index}]")
                    for index, item in enumerate(value)
                ]
            }
        if isinstance(value, (set, frozenset)):
            encoded_items = [
                self._encode_value(item, path=f"{path}{{set}}")
                for item in value
            ]
            encoded_items.sort(
                key=lambda item: json.dumps(
                    item, sort_keys=True, separators=(",", ":")
                )
            )
            return {"$set": encoded_items}
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError(f"dictionary keys at {path} must be strings")
            return {
                "$dict": {
                    key: self._encode_value(item, path=f"{path}[{key!r}]")
                    for key, item in value.items()
                }
            }
        if is_dataclass(value) and not isinstance(value, type):
            tag = _type_tag(type(value))
            if tag not in self._types:
                raise TypeError(
                    f"nested dataclass type {tag!r} is not registered"
                )
            return {
                "$dataclass": tag,
                "fields": {
                    field.name: self._encode_value(
                        getattr(value, field.name),
                        path=f"{path}.{field.name}",
                    )
                    for field in fields(value)
                    if not (field.metadata.get("omit_none") and getattr(value, field.name) is None)
                },
            }
        raise TypeError(
            f"unsupported dataclass field value at {path}: "
            f"{type(value).__module__}.{type(value).__qualname__}; "
            "numeric arrays must be stored as Blob references"
        )

    def _decode_value(self, value: Any, *, path: str) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"non-finite float at {path}")
            return value
        if not isinstance(value, dict):
            raise ValueError(f"invalid encoded value at {path}")

        if set(value) == {"$bytes"}:
            encoded = value["$bytes"]
            if not isinstance(encoded, str):
                raise ValueError(f"invalid bytes value at {path}")
            return base64.b64decode(encoded.encode("ascii"), validate=True)
        if set(value) == {"$path"}:
            encoded = value["$path"]
            if not isinstance(encoded, str):
                raise ValueError(f"invalid path value at {path}")
            return Path(encoded)
        if set(value) == {"$tuple"}:
            encoded = value["$tuple"]
            if not isinstance(encoded, list):
                raise ValueError(f"invalid tuple value at {path}")
            return tuple(
                self._decode_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(encoded)
            )
        if set(value) == {"$list"}:
            encoded = value["$list"]
            if not isinstance(encoded, list):
                raise ValueError(f"invalid list value at {path}")
            return [
                self._decode_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(encoded)
            ]
        if set(value) == {"$set"}:
            encoded = value["$set"]
            if not isinstance(encoded, list):
                raise ValueError(f"invalid set value at {path}")
            return {
                self._decode_value(item, path=f"{path}{{set}}")
                for item in encoded
            }
        if set(value) == {"$dict"}:
            encoded = value["$dict"]
            if not isinstance(encoded, dict):
                raise ValueError(f"invalid dictionary value at {path}")
            if any(not isinstance(key, str) for key in encoded):
                raise ValueError(f"invalid dictionary key at {path}")
            return {
                key: self._decode_value(item, path=f"{path}[{key!r}]")
                for key, item in encoded.items()
            }
        if set(value) == {"$dataclass", "fields"}:
            tag = value["$dataclass"]
            encoded_fields = value["fields"]
            if not isinstance(tag, str) or not isinstance(encoded_fields, dict):
                raise ValueError(f"invalid dataclass envelope at {path}")
            value_type = self._types.get(tag)
            if value_type is None:
                raise TypeError(f"unregistered dataclass type {tag!r}")
            expected_fields = {field.name for field in fields(value_type)}
            optional_fields = {
                field.name for field in fields(value_type)
                if field.metadata.get("omit_none") and field.default is None
            }
            if not expected_fields - optional_fields <= set(encoded_fields) <= expected_fields:
                raise ValueError(
                    f"dataclass fields for {tag!r} do not match the registered type"
                )
            kwargs = {
                name: self._decode_value(item, path=f"{path}.{name}")
                for name, item in encoded_fields.items()
            }
            return value_type(**kwargs)
        raise ValueError(f"unknown dataclass codec envelope at {path}")

    def _collect_references(self, value: Any, *, path: str) -> list[Blob]:
        if value is None or isinstance(value, (str, bool, int, bytes, Path)):
            return []
        if isinstance(value, float):
            # Typed-root reference discovery historically accepts all floats.
            if self.value_type is object and not math.isfinite(value):
                raise ValueError(f"non-finite float at {path}")
            return []
        if isinstance(value, Blob):
            if self.value_type is object:
                # A Blob is a reference leaf, but its fields still need the
                # object-root structural validation (including its allowlist).
                self._validate_structured_value(value, path=path)
            return [value]
        if isinstance(value, (tuple, list, set, frozenset)):
            result: list[Blob] = []
            for index, item in enumerate(value):
                result.extend(
                    self._collect_references(item, path=f"{path}[{index}]")
                )
            return result
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError(f"dictionary keys at {path} must be strings")
            result = []
            for key, item in value.items():
                result.extend(
                    self._collect_references(item, path=f"{path}[{key!r}]")
                )
            return result
        if is_dataclass(value) and not isinstance(value, type):
            tag = _type_tag(type(value))
            if tag not in self._types:
                raise TypeError(
                    f"nested dataclass type {tag!r} is not registered"
                )
            result = []
            for field in fields(value):
                result.extend(
                    self._collect_references(
                        getattr(value, field.name),
                        path=f"{path}.{field.name}",
                    )
                )
            return result
        raise TypeError(
            f"unsupported referenced value at {path}: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )


def _type_tag(value_type: type[Any]) -> str:
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _design_dataclass_types() -> tuple[type[Any], ...]:
    result: list[type[Any]] = []
    for name in getattr(design_types, "__all__", ()):
        value = getattr(design_types, name, None)
        if (
            isinstance(value, type)
            and is_dataclass(value)
            and value.__module__ == design_types.__name__
        ):
            result.append(value)
    return tuple(result)


def _file_identity(status: os.stat_result) -> tuple[int, ...]:
    # Do not use atime: reading a blob must not invalidate its proof.
    return (status.st_dev, status.st_ino, status.st_mode, status.st_size,
            status.st_mtime_ns, status.st_ctime_ns)


@dataclass(frozen=True)
class _DurableMember:
    encoded: bytes
    codec: Codec[Any]
    compatibility: Any
    children: tuple[Blob, ...]
    file_identity: tuple[int, ...]
    directories: tuple[tuple[Path, tuple[int, int]], ...]


class DataStore:
    """Content-addressed storage with memory-only and optional durable modes."""

    # Bound retained proofs, not the existing store's scientific data lifetime.
    _PROOF_LIMIT = 4096

    def __init__(
        self, root: Path | None = None, *, reuse_durable_closures: bool = False,
    ) -> None:
        self.root = None if root is None else Path(root)
        self._memory: dict[Blob, bytes] = {}
        self._codecs: dict[str, Codec[Any]] = {}
        self._durable_members: OrderedDict[Blob, _DurableMember] = OrderedDict()
        self._durable_directories: OrderedDict[Path, tuple[int, ...]] = OrderedDict()
        self._durable_scope: tuple[int, Path] | None = None
        self.reuse_durable_closures = reuse_durable_closures

    @property
    def reuse_durable_closures(self) -> bool:
        return self._reuse_durable_closures

    @reuse_durable_closures.setter
    def reuse_durable_closures(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("reuse_durable_closures must be bool")
        if enabled != getattr(self, "_reuse_durable_closures", None):
            self._clear_durable_proofs()
        self._reuse_durable_closures = enabled

    def _clear_durable_proofs(self) -> None:
        self._durable_members.clear()
        self._durable_directories.clear()
        self._durable_scope = None

    def register(self, codec: Codec[Any]) -> Codec[Any]:
        identity = getattr(codec, "identity", None)
        if not isinstance(identity, str) or not identity:
            raise TypeError("codec must expose a non-empty string identity")
        for method in ("encode", "decode", "references"):
            if not callable(getattr(codec, method, None)):
                raise TypeError(f"codec must provide {method}()")
        existing = self._codecs.get(identity)
        if existing is not None and existing is not codec:
            existing_key = _codec_compatibility(existing)
            new_key = _codec_compatibility(codec)
            if existing_key != new_key:
                raise ValueError(
                    f"codec identity {identity!r} is already registered with "
                    "an incompatible implementation"
                )
        self._codecs.setdefault(identity, codec)
        return codec

    @profiled("datastore.put")
    def put(self, value: Any, codec: Codec[Any]) -> Blob:
        self.register(codec)
        if self.root is not None and getattr(codec, "persistent", True) is False:
            raise ValueError("Process-local resources cannot enter a durable DataStore")
        encoded = codec.encode(value)
        if not isinstance(encoded, bytes):
            raise TypeError("codec.encode() must return bytes")
        references = codec.references(value)
        if not isinstance(references, tuple) or any(
            not isinstance(reference, Blob) for reference in references
        ):
            raise TypeError("codec.references() must return tuple[Blob, ...]")
        reference = Blob(
            identity=Key(hashlib.sha256(encoded).hexdigest()),
            codec=codec.identity,
        )
        existing = self._memory.get(reference)
        if existing is not None and existing != encoded:
            raise ValueError("codec produced different bytes for the same digest")
        self._memory[reference] = encoded
        return reference

    @profiled("datastore.get")
    def get(
        self,
        reference: Blob,
        codec: Codec[T] | None = None,
    ) -> T:
        if not isinstance(reference, Blob):
            raise TypeError("reference must be a Blob")
        if codec is not None:
            self.register(codec)
            if codec.identity != reference.codec:
                raise ValueError(
                    f"reference codec {reference.codec!r} does not match "
                    f"supplied codec {codec.identity!r}"
                )
        registered = self._codecs.get(reference.codec)
        if registered is None:
            raise KeyError(f"unknown codec {reference.codec!r}")
        if codec is not None and registered is not codec:
            if _codec_compatibility(registered) != _codec_compatibility(codec):
                raise ValueError("registered codec is incompatible with supplied codec")

        encoded = self._load_bytes(reference)
        digest = hashlib.sha256(encoded).hexdigest()
        if digest != reference.identity.digest:
            raise ValueError(
                f"content digest mismatch for {reference.codec}:"
                f"{reference.identity.digest}"
            )
        value = registered.decode(encoded)
        references = registered.references(value)
        if not isinstance(references, tuple) or any(
            not isinstance(item, Blob) for item in references
        ):
            raise TypeError("codec.references() must return tuple[Blob, ...]")
        return value

    @profiled("datastore.persist")
    def persist(self, references: tuple[Blob, ...]) -> None:
        """Validate and durably write the complete referenced-closure."""

        if self.root is None:
            raise RuntimeError("memory-only DataStore cannot persist")
        if not isinstance(references, tuple) or any(
            not isinstance(reference, Blob) for reference in references
        ):
            raise TypeError("references must be tuple[Blob, ...]")

        if self.reuse_durable_closures:
            self._persist_reusing(references)
            return

        queue = list(references)
        seen: set[Blob] = set()
        encoded_by_reference: dict[Blob, bytes] = {}
        while queue:
            reference = queue.pop(0)
            if reference in seen:
                continue
            seen.add(reference)
            codec = self._codecs.get(reference.codec)
            if codec is None:
                raise KeyError(
                    f"cannot persist unknown codec {reference.codec!r}"
                )
            if getattr(codec, "persistent", True) is False:
                raise ValueError("Process-local resources cannot be persisted, including through another value")
            encoded = self._load_bytes(reference)
            digest = hashlib.sha256(encoded).hexdigest()
            if digest != reference.identity.digest:
                raise ValueError(
                    f"content digest mismatch for {reference.codec}:"
                    f"{reference.identity.digest}"
                )
            value = codec.decode(encoded)
            child_references = codec.references(value)
            if not isinstance(child_references, tuple) or any(
                not isinstance(item, Blob) for item in child_references
            ):
                raise TypeError("codec.references() must return tuple[Blob, ...]")
            encoded_by_reference[reference] = encoded
            queue.extend(child_references)

        for reference, encoded in encoded_by_reference.items():
            self._write_reference(reference, encoded)
        # A journal may only reference names whose directory entries are also
        # durable. fsync(file) alone does not make a subsequent rename durable.
        directories = set()
        for reference in encoded_by_reference:
            directory = self._path_for(reference).parent
            while directory != self.root.parent:
                directories.add(directory)
                directory = directory.parent
        directories.add(self.root.parent)
        for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _persist_reusing(self, references: tuple[Blob, ...]) -> None:
        # Forks, root changes and cwd changes for relative roots start cold.
        scope = (os.getpid(), self.root.absolute())
        if scope != self._durable_scope:
            self._clear_durable_proofs()
            self._durable_scope = scope
        queue = deque(references)
        seen: set[Blob] = set()
        pending: dict[Blob, tuple[bytes, Codec[Any], Any, tuple[Blob, ...]]] = {}
        members: dict[Blob, _DurableMember] = {}
        directory_status: dict[Path, os.stat_result] = {}
        checked = served = unsupported = 0

        def directory_stat(path: Path) -> os.stat_result:
            if path not in directory_status:
                directory_status[path] = path.lstat()
            return directory_status[path]

        while queue:
            reference = queue.popleft()
            if reference in seen:
                continue
            seen.add(reference)
            codec = self._codecs.get(reference.codec)
            if codec is None:
                self._durable_members.pop(reference, None)
                raise KeyError(f"cannot persist unknown codec {reference.codec!r}")
            if getattr(codec, "persistent", True) is False:
                self._durable_members.pop(reference, None)
                raise ValueError("Process-local resources cannot be persisted, including through another value")
            # A custom codec's compatibility key need not describe mutable
            # decode/references behavior. Only these exact built-ins have the
            # known byte-to-closure contract; subclasses must also revalidate.
            reusable = type(codec) in (JSONCodec, ArrayCodec, DataclassCodec)
            if not reusable:
                unsupported += 1
                self._durable_members.pop(reference, None)
            compatibility = _codec_compatibility(codec) if reusable else None
            proof = self._durable_members.get(reference)
            if proof is not None:
                checked += 1
                memory = self._memory.get(reference)
                valid = (
                    codec is proof.codec and compatibility == proof.compatibility
                    and (memory is None or (
                        isinstance(memory, bytes) and memory == proof.encoded
                    ))
                )
                if valid:
                    try:
                        valid = (
                            _file_identity(self._path_for(reference).lstat())
                            == proof.file_identity
                            and all(
                                (directory_stat(path).st_dev, directory_stat(path).st_ino)
                                == identity for path, identity in proof.directories
                            )
                        )
                    except FileNotFoundError:
                        valid = False
                if valid:
                    served += 1
                    self._durable_members.move_to_end(reference)
                    members[reference] = proof
                    queue.extend(proof.children)
                    continue
                self._durable_members.pop(reference, None)

            # A proof is not a fallback source of bytes: the strict path still
            # decides whether a missing file can be restored from live memory.
            encoded = self._load_bytes(reference)
            if hashlib.sha256(encoded).hexdigest() != reference.identity.digest:
                raise ValueError(
                    f"content digest mismatch for {reference.codec}:"
                    f"{reference.identity.digest}"
                )
            value = codec.decode(encoded)
            children = codec.references(value)
            if not isinstance(children, tuple) or any(
                not isinstance(child, Blob) for child in children
            ):
                raise TypeError("codec.references() must return tuple[Blob, ...]")
            pending[reference] = (encoded, codec, deepcopy(compatibility), children)
            queue.extend(children)

        # Finish validating the whole closure before changing any files.
        identities = {}
        for reference, (encoded, codec, _, _) in pending.items():
            identities[reference] = self._write_reference(
                reference, encoded,
                prove_durable=type(codec) in (JSONCodec, ArrayCodec, DataclassCodec),
            )
        if pending:
            # New directory entries can change ancestor timestamps. Never seal
            # a directory proof using a pre-write stat snapshot.
            directory_status.clear()
        directories = {scope[1].parent}
        for reference in seen:
            if reference in members:
                directories.update(path for path, _ in members[reference].directories)
                continue
            paths = []
            directory = self._path_for(reference).absolute().parent
            while directory != scope[1].parent:
                paths.append(directory)
                directory = directory.parent
            paths.append(directory)
            directories.update(paths)
            encoded, codec, compatibility, children = pending[reference]
            identity = identities[reference]
            if identity is not None:
                members[reference] = _DurableMember(
                    encoded, codec, compatibility, children, identity,
                    tuple((path, (directory_stat(path).st_dev, directory_stat(path).st_ino))
                          for path in paths),
                )

        synced = {}
        for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
            identity = _file_identity(directory_stat(directory))
            if self._durable_directories.get(directory) == identity:
                self._durable_directories.move_to_end(directory)
                continue
            self._durable_directories.pop(directory, None)
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                before = _file_identity(os.fstat(descriptor))
                os.fsync(descriptor)
                after = _file_identity(os.fstat(descriptor))
                if before == after == identity == _file_identity(directory.lstat()):
                    synced[directory] = identity
            finally:
                os.close(descriptor)

        # No proof is published until the complete file/directory barrier has
        # succeeded. Eviction only costs work; it never changes stored data.
        for reference, proof in members.items():
            if not all(
                path in synced or path in self._durable_directories
                for path, _ in proof.directories
            ):
                # Symlinks and concurrent namespace changes cannot acquire a
                # directory proof from fsync of a different underlying inode.
                self._durable_members.pop(reference, None)
                continue
            self._durable_members[reference] = proof
            self._durable_members.move_to_end(reference)
        self._durable_directories.update(synced)
        for cache in (self._durable_members, self._durable_directories):
            while len(cache) > self._PROOF_LIMIT:
                cache.popitem(last=False)
        event("datastore.persist.reuse", checked=checked, served=served,
              miss=len(pending), unsupported=unsupported, members=len(seen),
              directories_synced=len(synced))

    def validate_chemical_edit(self, system, variable, old, new):
        """Validate graph edits at their owner against the represented maps.

        Atom insertion/deletion or connectivity changes cannot silently retain
        a System's old occurrence and attachment maps.
        """
        if old.codec != new.codec:
            raise ValueError("Chemical edits require the same graph codec owner")
        codec = self._codecs.get(old.codec)
        validator = getattr(codec, "validate_edit", None)
        if not callable(validator):
            raise ValueError("Chemical graph owner does not validate mapped edits")
        old_value, new_value = self.get(old), self.get(new)
        topology = self.get(system.topology)
        maps = tuple(self.get(o.index_map) for o in system.occurrences if o.variable == variable)
        if validator(old_value, new_value, topology=topology, occurrence_maps=maps) is not True:
            raise ValueError("Chemical graph owner refused the represented edit")

    def validate_chemical_reference(
        self,
        reference: Blob,
        fixed_atoms: tuple[str, ...],
    ) -> Any:
        """Delegate canonical fixed-atom validation to the graph codec owner.

        The owner method must include incident bonds, bond orders, stereo,
        charge, attachments, and any changed cross-boundary bond data in the
        returned canonical records.
        """

        if not isinstance(reference, Blob):
            raise TypeError("chemical graph reference must be a Blob")
        codec = self._codecs.get(reference.codec)
        if codec is None:
            raise KeyError(f"unknown chemical graph codec {reference.codec!r}")
        validator = getattr(codec, "fixed_atom_records", None)
        if not callable(validator):
            raise ValueError(
                f"codec {reference.codec!r} does not provide canonical "
                "chemical fixed_atom_records() validation"
            )
        value = self.get(reference, codec)
        return validator(value, tuple(fixed_atoms))

    def _load_bytes(self, reference: Blob) -> bytes:
        encoded = self._memory.get(reference)
        if encoded is not None:
            return encoded
        if self.root is None:
            raise KeyError(
                f"blob {reference.codec}:{reference.identity.digest} is not in memory"
            )
        path = self._path_for(reference)
        if not path.is_file():
            raise KeyError(
                f"blob {reference.codec}:{reference.identity.digest} is unavailable"
            )
        return path.read_bytes()

    def _path_for(self, reference: Blob) -> Path:
        if self.root is None:
            raise RuntimeError("memory-only DataStore has no paths")
        codec_dir = quote(reference.codec, safe="")
        digest = reference.identity.digest
        return self.root / codec_dir / digest[:2] / digest

    def _write_reference(
        self, reference: Blob, encoded: bytes, *, prove_durable: bool = False,
    ) -> tuple[int, ...] | None:
        target = self._path_for(reference)
        try:
            with target.open("rb") as handle:
                before = _file_identity(os.fstat(handle.fileno())) if prove_durable else None
                existing = handle.read()
                if existing != encoded:
                    raise ValueError(
                        f"durable content differs at {reference.codec}:"
                        f"{reference.identity.digest}"
                    )
                if prove_durable:
                    # Existing bytes are not evidence of a prior file fsync.
                    os.fsync(handle.fileno())
                    after = _file_identity(os.fstat(handle.fileno()))
                    if before == after == _file_identity(target.lstat()):
                        return after
                return None
        except FileNotFoundError:
            # Only a missing target needs directory creation. Existing closure
            # members still have their actual bytes checked on every persist.
            target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
                before = os.fstat(handle.fileno()) if prove_durable else None
                os.replace(temporary_name, target)
                if prove_durable:
                    after = os.fstat(handle.fileno())
                    # Rename may change ctime, but not the synced contents.
                    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (
                        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                    ) and _file_identity(after) == _file_identity(target.lstat()):
                        return _file_identity(after)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _codec_compatibility(codec: Codec[Any]) -> Any:
    compatibility = getattr(codec, "compatibility_key", None)
    if callable(compatibility):
        return compatibility()
    return (type(codec).__module__, type(codec).__qualname__, codec.identity)


design_codec = DataclassCodec(object, "astrolabe.design/v1")


__all__ = [
    "ArrayCodec",
    "Codec",
    "DataStore",
    "DataclassCodec",
    "JSONCodec",
    "design_codec",
]
