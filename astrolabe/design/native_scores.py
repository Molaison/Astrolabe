"""Persistent native ``StandardScores`` compatibility envelope.

This is an owner-local transitional format, not a generic operation payload.
It snapshots the already-normalized native result, references numeric arrays
through ``DataStore``, and carries selected structure bytes separately from
their original path provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
import json
import math
import os
from pathlib import Path, PurePath
import re
from typing import Any

from .store import ArrayCodec, DataStore, DataclassCodec
from .types import Blob, Key


_SCHEMA = "astrolabe.design.native-scores-envelope/v2"
_FIELDS_SCHEMA = "astrolabe.design.native-scores-fields/v2"


class UnsupportedNativeScoresState(ValueError):
    """Native score state cannot be persisted without an exact owner codec."""


@contextmanager
def _active_guard(value: Any, active: set[int]):
    identity = id(value)
    if identity in active:
        raise UnsupportedNativeScoresState(
            "native score extensions contain a reference cycle"
        )
    active.add(identity)
    try:
        yield
    finally:
        active.remove(identity)


def _validate_extension_name(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise UnsupportedNativeScoresState(
            "native score extension attribute names must be non-empty strings"
        )
    if not name.isidentifier():
        raise UnsupportedNativeScoresState(
            f"native score extension attribute {name!r} is not an identifier"
        )
    if name.startswith("__") and name.endswith("__"):
        raise UnsupportedNativeScoresState(
            f"native score extension attribute {name!r} is reserved"
        )
    from astrolabe.evaluators.utils import StandardScores

    if hasattr(StandardScores, name):
        raise UnsupportedNativeScoresState(
            f"native score extension attribute {name!r} collides with "
            "StandardScores"
        )
    return name


class NativeScoresFieldCodec:
    """Canonical JSON codec for tagged field values and array references."""

    identity = _FIELDS_SCHEMA

    def encode(self, value: Any) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    def decode(self, data: bytes) -> Any:
        if not isinstance(data, bytes):
            raise TypeError("encoded native score fields must be bytes")
        return json.loads(
            data.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value!r}")
            ),
        )

    def references(self, value: Any) -> tuple[Blob, ...]:
        result: list[Blob] = []

        def visit(item: Any) -> None:
            if isinstance(item, list):
                for child in item:
                    visit(child)
                return
            if not isinstance(item, dict):
                return
            if set(item) == {"$array"}:
                record = item["$array"]
                if not isinstance(record, dict) or set(record) != {"codec", "digest"}:
                    raise ValueError("invalid native score array reference")
                codec = record["codec"]
                digest = record["digest"]
                if not isinstance(codec, str) or not codec:
                    raise ValueError("invalid native score array codec")
                if not isinstance(digest, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", digest
                ):
                    raise ValueError("invalid native score array digest")
                result.append(Blob(Key(digest), codec))
                return
            if set(item) == {"$native_scores"}:
                record = item["$native_scores"]
                if not isinstance(record, dict) or set(record) != {
                    "codec",
                    "digest",
                }:
                    raise ValueError("invalid nested native score reference")
                codec = record["codec"]
                digest = record["digest"]
                if codec != NATIVE_SCORES_ENVELOPE_CODEC.identity:
                    raise ValueError(
                        f"unexpected nested native score codec {codec!r}"
                    )
                if not isinstance(digest, str) or not re.fullmatch(
                    r"[0-9a-f]{64}", digest
                ):
                    raise ValueError("invalid nested native score digest")
                result.append(Blob(Key(digest), codec))
                return
            for child in item.values():
                visit(child)

        visit(value)
        return tuple(result)

    def compatibility_key(self) -> tuple[str, str]:
        return (type(self).__name__, self.identity)


class NativeStructureBytesCodec:
    """Owner codec for immutable selected-structure bytes."""

    identity = "astrolabe.design.native-structure-bytes/v1"

    def encode(self, value: bytes) -> bytes:
        if not isinstance(value, bytes):
            raise TypeError("native structure content must be bytes")
        return value

    def decode(self, data: bytes) -> bytes:
        if not isinstance(data, bytes):
            raise TypeError("encoded native structure content must be bytes")
        return bytes(data)

    def references(self, value: bytes) -> tuple[Blob, ...]:
        if not isinstance(value, bytes):
            raise TypeError("native structure content must be bytes")
        return ()

    def compatibility_key(self) -> tuple[str, str]:
        return (type(self).__name__, self.identity)


@dataclass(frozen=True)
class NativeScoreField:
    name: str
    value: Blob


@dataclass(frozen=True)
class NativeSourceStructure:
    assembly: str
    original_path: str
    restored_name: str
    content: Blob


@dataclass(frozen=True)
class NativeScoresEnvelope:
    schema: str
    fields: tuple[NativeScoreField, ...]
    structures: tuple[NativeSourceStructure, ...]


NATIVE_SCORES_FIELD_CODEC = NativeScoresFieldCodec()
NATIVE_STRUCTURE_BYTES_CODEC = NativeStructureBytesCodec()
NATIVE_SCORES_ENVELOPE_CODEC = DataclassCodec(
    NativeScoresEnvelope,
    _SCHEMA,
    nested_types=(NativeScoreField, NativeSourceStructure),
)
NATIVE_SCORES_REFERENCE_CODECS = (
    NATIVE_SCORES_FIELD_CODEC,
    NATIVE_SCORES_ENVELOPE_CODEC,
    NATIVE_STRUCTURE_BYTES_CODEC,
)


def _blob_record(blob: Blob) -> dict[str, str]:
    return {"codec": blob.codec, "digest": blob.identity.digest}


def _float_record(value: float) -> float | dict[str, str]:
    if math.isnan(value):
        return {"$nonfinite_float": "nan"}
    if math.isinf(value):
        return {"$nonfinite_float": "inf" if value > 0 else "-inf"}
    return value


def _encode_value(
    value: Any,
    data: DataStore,
    active: set[int],
) -> Any:
    if value is None:
        return {"$none": True}
    if type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        return _float_record(value)
    if isinstance(value, PurePath):
        path_types = {
            "PosixPath",
            "PurePosixPath",
            "WindowsPath",
            "PureWindowsPath",
        }
        type_name = type(value).__name__
        if type_name not in path_types:
            raise UnsupportedNativeScoresState(
                f"unsupported native score path type: {type_name}"
            )
        return {
            "$path": {
                "type": type_name,
                "value": str(value),
            }
        }

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if type(value) is not np.ndarray or value.dtype.metadata:
                raise UnsupportedNativeScoresState(
                    "score array subclass/metadata requires an explicit native codec"
                )
            if value.dtype.hasobject:
                raise UnsupportedNativeScoresState(
                    "object-dtype score arrays have no safe native codec"
                )
            return {
                "$array": _blob_record(data.put(value, ArrayCodec())),
            }
        if isinstance(value, np.generic):
            if value.dtype.fields or value.dtype.metadata:
                raise UnsupportedNativeScoresState(
                    "structured/metadata numpy scalars require an explicit native codec"
                )
            if value.dtype.hasobject:
                raise UnsupportedNativeScoresState(
                    "object-dtype numpy scalars are unsupported"
                )
            return {
                "$numpy_scalar": {
                    "dtype": value.dtype.str,
                    "value": _encode_value(value.item(), data, active),
                }
            }
    except ImportError:
        pass

    if type(value) is list:
        with _active_guard(value, active):
            return {
                "$list": [_encode_value(item, data, active) for item in value],
            }
    if type(value) is tuple:
        with _active_guard(value, active):
            return {
                "$tuple": [_encode_value(item, data, active) for item in value],
            }
    if type(value) is dict:
        with _active_guard(value, active):
            entries = []
            for key, item in value.items():
                if not isinstance(key, str):
                    raise UnsupportedNativeScoresState(
                        "native score mapping keys must be strings"
                    )
                entries.append([key, _encode_value(item, data, active)])
            return {"$dict": entries}

    from astrolabe.evaluators.utils import StandardScores

    if type(value) is StandardScores:
        nested_blob = snapshot_scores(value, data, _active=active)
        return {"$native_scores": _blob_record(nested_blob)}

    from astrolabe.score_contributions import ScoreContribution

    if type(value) is ScoreContribution:
        with _active_guard(value, active):
            return {
                "$score_contribution": {
                    field.name: _encode_value(
                        getattr(value, field.name),
                        data,
                        active,
                    )
                    for field in fields(ScoreContribution)
                }
            }
    if is_dataclass(value):
        if type(value) in _evaluator_evidence_types().values():
            with _active_guard(value, active):
                return {
                    "$evaluator_evidence": {
                        "type": type(value).__name__,
                        "fields": {
                            field.name: _encode_value(getattr(value, field.name), data, active)
                            for field in fields(value)
                        },
                    }
                }
        raise UnsupportedNativeScoresState(
            f"unsupported native score dataclass: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )
    raise UnsupportedNativeScoresState(
        "unsupported native score value: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _evaluator_evidence_types():
    from astrolabe.evaluators.evidence import (
        CandidateGroup, DerivedStructure, EvaluationRecord, InputObservation,
    )

    return {cls.__name__: cls for cls in (
        CandidateGroup, DerivedStructure, EvaluationRecord, InputObservation,
    )}


def _decode_nonfinite(value: Any) -> float:
    if value == "nan":
        return math.nan
    if value == "inf":
        return math.inf
    if value == "-inf":
        return -math.inf
    raise ValueError(f"unknown non-finite float marker {value!r}")


def _decode_path(record: Any) -> PurePath:
    if not isinstance(record, dict) or set(record) != {"type", "value"}:
        raise ValueError("invalid native score path value")
    type_name = record["type"]
    value = record["value"]
    if not isinstance(type_name, str) or not isinstance(value, str):
        raise ValueError("native score path type/value must be strings")
    from pathlib import PurePosixPath, PureWindowsPath

    factories = {
        "PosixPath": Path,
        "PurePosixPath": PurePosixPath,
        "WindowsPath": Path,
        "PureWindowsPath": PureWindowsPath,
    }
    factory = factories.get(type_name)
    if factory is None:
        raise ValueError(f"unsupported native score path type {type_name!r}")
    try:
        path = factory(value)
    except (NotImplementedError, ValueError) as exc:
        raise ValueError(
            f"cannot restore native score path type {type_name!r}"
        ) from exc
    if type(path).__name__ != type_name:
        raise ValueError(
            f"cannot preserve native score path type {type_name!r} "
            f"on this platform"
        )
    return path


def _decode_value(
    value: Any,
    data: DataStore,
    destination: Path,
    active_blobs: set[Blob],
) -> Any:
    if value is None or type(value) in {bool, int, str, float}:
        return value
    if not isinstance(value, dict):
        raise ValueError("invalid tagged native score value")

    if set(value) == {"$none"} and value["$none"] is True:
        return None
    if set(value) == {"$nonfinite_float"}:
        return _decode_nonfinite(value["$nonfinite_float"])
    if set(value) == {"$path"}:
        return _decode_path(value["$path"])
    if set(value) == {"$array"}:
        record = value["$array"]
        if not isinstance(record, dict) or set(record) != {"codec", "digest"}:
            raise ValueError("invalid native score array reference")
        if record["codec"] != ArrayCodec().identity:
            raise ValueError(
                f"unexpected native score array codec {record['codec']!r}"
            )
        blob = Blob(Key(record["digest"]), record["codec"])
        array = data.get(blob, ArrayCodec())
        import numpy as np

        return np.array(array, copy=True)
    if set(value) == {"$numpy_scalar"}:
        record = value["$numpy_scalar"]
        if not isinstance(record, dict) or set(record) != {"dtype", "value"}:
            raise ValueError("invalid native numpy scalar")
        import numpy as np

        return np.array(
            _decode_value(
                record["value"],
                data,
                destination,
                active_blobs,
            ),
            dtype=np.dtype(record["dtype"]),
        )[()]
    if set(value) == {"$list"}:
        items = value["$list"]
        if not isinstance(items, list):
            raise ValueError("invalid native list value")
        return [
            _decode_value(item, data, destination, active_blobs)
            for item in items
        ]
    if set(value) == {"$tuple"}:
        items = value["$tuple"]
        if not isinstance(items, list):
            raise ValueError("invalid native tuple value")
        return tuple(
            _decode_value(item, data, destination, active_blobs)
            for item in items
        )
    if set(value) == {"$dict"}:
        entries = value["$dict"]
        if not isinstance(entries, list):
            raise ValueError("invalid native mapping value")
        result: dict[str, Any] = {}
        for entry in entries:
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or entry[0] in result
            ):
                raise ValueError("invalid or duplicate native mapping entry")
            result[entry[0]] = _decode_value(
                entry[1], data, destination, active_blobs
            )
        return result
    if set(value) == {"$native_scores"}:
        record = value["$native_scores"]
        if not isinstance(record, dict) or set(record) != {"codec", "digest"}:
            raise ValueError("invalid nested native score reference")
        codec = record["codec"]
        digest = record["digest"]
        if codec != NATIVE_SCORES_ENVELOPE_CODEC.identity:
            raise ValueError(f"unexpected nested native score codec {codec!r}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid nested native score digest")
        nested_blob = Blob(Key(digest), codec)
        nested_destination = destination / "_native_scores" / digest
        return restore_scores(
            nested_blob,
            data,
            nested_destination,
            _active_blobs=active_blobs,
        )
    if set(value) == {"$score_contribution"}:
        record = value["$score_contribution"]
        if not isinstance(record, dict):
            raise ValueError("invalid native score contribution")
        from astrolabe.score_contributions import ScoreContribution

        expected = {field.name for field in fields(ScoreContribution)}
        if set(record) != expected:
            raise ValueError("native score contribution fields changed")
        return ScoreContribution(
            **{
                name: _decode_value(
                    item, data, destination, active_blobs
                )
                for name, item in record.items()
            }
        )
    if set(value) == {"$evaluator_evidence"}:
        record = value["$evaluator_evidence"]
        if not isinstance(record, dict) or set(record) != {"type", "fields"}:
            raise ValueError("invalid evaluator evidence record")
        kind = record["type"]
        factory = _evaluator_evidence_types().get(kind) if isinstance(kind, str) else None
        if factory is None:
            raise ValueError("unknown evaluator evidence type")
        members = record["fields"]
        if not isinstance(members, dict) or set(members) != {field.name for field in fields(factory)}:
            raise ValueError("evaluator evidence fields changed")
        return factory(**{
            name: _decode_value(item, data, destination, active_blobs)
            for name, item in members.items()
        })
    raise ValueError("unknown tagged native score value")


def _safe_restored_name(assembly: str, original: Path) -> str:
    basename = original.name
    if not basename or basename in {".", ".."}:
        basename = "structure"
    safe_assembly = re.sub(r"[^A-Za-z0-9_.-]+", "_", assembly).strip("._")
    if not safe_assembly:
        safe_assembly = "assembly"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", basename).strip("._")
    return f"{safe_assembly}-{safe_name or 'structure'}"


def _encode_complex_paths(
    value: Any,
    data: DataStore,
    structures: list[NativeSourceStructure],
) -> dict[str, Any]:
    if value is None:
        return {"$none": True}
    if not isinstance(value, Mapping):
        raise UnsupportedNativeScoresState(
            "complex_paths must be a mapping of assembly to structure path"
        )
    entries: list[list[Any]] = []
    restored_names: set[str] = set()
    for assembly, raw_path in value.items():
        if not isinstance(assembly, str) or not assembly:
            raise UnsupportedNativeScoresState(
                "complex_paths assembly names must be non-empty strings"
            )
        if not isinstance(raw_path, (str, os.PathLike)):
            raise UnsupportedNativeScoresState(
                f"complex path for {assembly!r} is not path-like"
            )
        source = Path(raw_path)
        if not source.is_file():
            raise UnsupportedNativeScoresState(
                f"complex path for {assembly!r} is unavailable: {source}"
            )
        restored_name = _safe_restored_name(assembly, source)
        if restored_name in restored_names:
            raise UnsupportedNativeScoresState(
                f"multiple structures map to restored name {restored_name!r}"
            )
        restored_names.add(restored_name)
        content = data.put(source.read_bytes(), NATIVE_STRUCTURE_BYTES_CODEC)
        structure_index = len(structures)
        structures.append(
            NativeSourceStructure(
                assembly=assembly,
                original_path=source.as_posix(),
                restored_name=restored_name,
                content=content,
            )
        )
        entries.append([assembly, structure_index])
    return {"$structure_paths": entries}


def _extension_items(
    scores: Any,
    declared_fields: tuple[str, ...],
) -> list[tuple[str, Any]]:
    state = vars(scores)
    missing = [name for name in declared_fields if name not in state]
    if missing:
        raise UnsupportedNativeScoresState(
            f"StandardScores is missing declared instance fields {missing}"
        )
    declared = set(declared_fields)
    extensions = []
    for name, value in state.items():
        if name in declared:
            continue
        _validate_extension_name(name)
        extensions.append((name, value))
    return sorted(extensions, key=lambda item: item[0])


def snapshot_scores(
    scores: Any,
    data: DataStore,
    *,
    _active: set[int] | None = None,
) -> Blob:
    """Freeze an exact score snapshot in DataStore and return its envelope.

    Persistence remains an explicit DataStore/Runtime operation.
    """

    if not isinstance(data, DataStore):
        raise TypeError("data must be a design DataStore")
    from astrolabe.evaluators.utils import StandardScores

    if type(scores) is not StandardScores:
        raise TypeError("scores must be a StandardScores instance")
    expected_fields = tuple(field.name for field in fields(StandardScores))
    actual_fields = tuple(field.name for field in fields(scores))
    if actual_fields != expected_fields:
        raise UnsupportedNativeScoresState(
            "StandardScores fields changed; update the native score envelope"
        )
    extensions = _extension_items(scores, expected_fields)
    if scores.complex_poses is not None and not (
        isinstance(scores.complex_poses, dict) and not scores.complex_poses
    ):
        raise UnsupportedNativeScoresState(
            "complex_poses contains live/arbitrary objects; a real pose codec is "
            "required before this score state can be persisted"
        )

    active = set() if _active is None else _active
    with _active_guard(scores, active):
        structures: list[NativeSourceStructure] = []
        rows: list[NativeScoreField] = []
        for field in fields(scores):
            value = getattr(scores, field.name)
            if field.name == "complex_paths":
                encoded = _encode_complex_paths(value, data, structures)
            else:
                encoded = _encode_value(value, data, active)
            rows.append(
                NativeScoreField(
                    name=field.name,
                    value=data.put(encoded, NATIVE_SCORES_FIELD_CODEC),
                )
            )
        for name, value in extensions:
            encoded = _encode_value(value, data, active)
            rows.append(
                NativeScoreField(
                    name=name,
                    value=data.put(encoded, NATIVE_SCORES_FIELD_CODEC),
                )
            )
        envelope = NativeScoresEnvelope(
            schema=_SCHEMA,
            fields=tuple(rows),
            structures=tuple(structures),
        )
    return data.put(envelope, NATIVE_SCORES_ENVELOPE_CODEC)


def _write_exact(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(
                f"restore destination already contains different data: {path}"
            )
        return
    descriptor = None
    created = False
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.is_file() and path.read_bytes() == content:
            return
        raise
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def _restore_structures(
    envelope: NativeScoresEnvelope,
    data: DataStore,
    destination: Path,
) -> tuple[Path, ...]:
    if envelope.schema != _SCHEMA:
        raise ValueError(f"unsupported native score envelope schema {envelope.schema!r}")
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        raise NotADirectoryError(destination)
    targets: list[Path] = []
    for structure in envelope.structures:
        content = data.get(structure.content, NATIVE_STRUCTURE_BYTES_CODEC)
        target = destination / structure.restored_name
        _write_exact(target, content)
        targets.append(target)
    return tuple(targets)


def _decode_complex_paths(
    value: Any,
    structures: tuple[NativeSourceStructure, ...],
    targets: tuple[Path, ...],
) -> Any:
    if value == {"$none": True}:
        return None
    if not isinstance(value, dict) or set(value) != {"$structure_paths"}:
        raise ValueError("invalid native complex_paths field")
    entries = value["$structure_paths"]
    if not isinstance(entries, list):
        raise ValueError("invalid native complex_paths entries")
    result: dict[str, Path] = {}
    used: set[int] = set()
    for entry in entries:
        if (
            not isinstance(entry, list)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            or type(entry[1]) is not int
            or entry[0] in result
            or entry[1] < 0
            or entry[1] >= len(structures)
        ):
            raise ValueError("invalid native complex_paths mapping")
        index = entry[1]
        if structures[index].assembly != entry[0]:
            raise ValueError("native complex path assembly provenance mismatch")
        used.add(index)
        result[entry[0]] = targets[index]
    if used != set(range(len(structures))):
        raise ValueError("native structure entries are not all referenced")
    return result


def _validate_field_layout(
    envelope: NativeScoresEnvelope,
    expected_fields: tuple[str, ...],
) -> tuple[NativeScoreField, ...]:
    names = tuple(row.name for row in envelope.fields)
    field_count = len(expected_fields)
    if names[:field_count] != expected_fields:
        raise UnsupportedNativeScoresState(
            "native score envelope field names/order do not match StandardScores"
        )
    extensions = envelope.fields[field_count:]
    extension_names = tuple(row.name for row in extensions)
    if len(set(names)) != len(names):
        raise UnsupportedNativeScoresState(
            "native score envelope contains colliding field names"
        )
    if extension_names != tuple(sorted(extension_names)):
        raise UnsupportedNativeScoresState(
            "native score extension fields are not sorted"
        )
    for name in extension_names:
        _validate_extension_name(name)
    return extensions


def _restore_scores(
    blob: Blob,
    data: DataStore,
    destination: Path,
    active_blobs: set[Blob],
) -> Any:
    envelope = data.get(blob, NATIVE_SCORES_ENVELOPE_CODEC)
    if not isinstance(envelope, NativeScoresEnvelope):
        raise TypeError("native score envelope decoded to an unexpected type")

    from astrolabe.evaluators.utils import StandardScores

    expected_fields = tuple(field.name for field in fields(StandardScores))
    extensions = _validate_field_layout(envelope, expected_fields)
    targets = _restore_structures(envelope, data, destination)
    values: dict[str, Any] = {}
    for row in envelope.fields:
        encoded = data.get(row.value, NATIVE_SCORES_FIELD_CODEC)
        if row.name == "complex_paths":
            value = _decode_complex_paths(
                encoded,
                envelope.structures,
                targets,
            )
        else:
            value = _decode_value(
                encoded,
                data,
                destination,
                active_blobs,
            )
        values[row.name] = value

    restored = object.__new__(StandardScores)
    for name in expected_fields:
        object.__setattr__(restored, name, values[name])
    for row in extensions:
        object.__setattr__(restored, row.name, values[row.name])
    return restored


def restore_scores(
    blob: Blob,
    data: DataStore,
    destination: Path | str,
    *,
    _active_blobs: set[Blob] | None = None,
) -> Any:
    """Restore exact native score fields and selected structure files."""

    if not isinstance(blob, Blob):
        raise TypeError("blob must be a Blob")
    if not isinstance(data, DataStore):
        raise TypeError("data must be a design DataStore")
    if blob.codec != NATIVE_SCORES_ENVELOPE_CODEC.identity:
        raise ValueError(
            f"blob codec {blob.codec!r} is not a native score envelope"
        )
    active_blobs = set() if _active_blobs is None else _active_blobs
    if blob in active_blobs:
        raise UnsupportedNativeScoresState(
            "native score references contain a cycle"
        )
    active_blobs.add(blob)
    try:
        return _restore_scores(blob, data, Path(destination), active_blobs)
    finally:
        active_blobs.remove(blob)


__all__ = [
    "NATIVE_SCORES_ENVELOPE_CODEC",
    "NATIVE_SCORES_FIELD_CODEC",
    "NATIVE_SCORES_REFERENCE_CODECS",
    "NATIVE_STRUCTURE_BYTES_CODEC",
    "NativeScoreField",
    "NativeScoresEnvelope",
    "NativeScoresFieldCodec",
    "NativeSourceStructure",
    "NativeStructureBytesCodec",
    "UnsupportedNativeScoresState",
    "restore_scores",
    "snapshot_scores",
]
