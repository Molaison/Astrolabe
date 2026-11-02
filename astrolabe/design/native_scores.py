"""Persistent native ``StandardScores`` compatibility envelope.

This is an owner-local transitional format, not a generic operation payload.
It snapshots the already-normalized native result, references numeric arrays
through ``DataStore``, and carries selected structure bytes separately from
their original path provenance.

Scalar NaNs retain NaN semantics, not their sign or payload bits. Restore callers
must own/control the destination namespace and exclude concurrent mutation of files
and ancestor directory entries. Final inode/content checks reject observed
replacement; ordinary returned Paths cannot prevent later external replacement.
Failed restores may leave partial files/directories: no name-based unlink is
safe against concurrent replacement, so cleanup is left to the directory owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, fields, is_dataclass
import errno
import json
import math
import os
from pathlib import Path, PurePath
import re
import stat
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
    ArrayCodec(),
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

        if isinstance(value, np.ma.MaskedArray):
            raise UnsupportedNativeScoresState(
                "MaskedArray score values are unsupported; a mask-preserving "
                "native codec is required"
            )
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise UnsupportedNativeScoresState(
                    "object-dtype score arrays have no safe native codec"
                )
            return {
                "$array": _blob_record(data.put(value, ArrayCodec())),
            }
        if isinstance(value, np.generic):
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
        raise UnsupportedNativeScoresState(
            f"unsupported native score dataclass: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )
    raise UnsupportedNativeScoresState(
        "unsupported native score value: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


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
    verification: _RestoreVerification,
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
                verification,
            ),
            dtype=np.dtype(record["dtype"]),
        )[()]
    if set(value) == {"$list"}:
        items = value["$list"]
        if not isinstance(items, list):
            raise ValueError("invalid native list value")
        return [
            _decode_value(item, data, destination, active_blobs, verification)
            for item in items
        ]
    if set(value) == {"$tuple"}:
        items = value["$tuple"]
        if not isinstance(items, list):
            raise ValueError("invalid native tuple value")
        return tuple(
            _decode_value(item, data, destination, active_blobs, verification)
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
                entry[1], data, destination, active_blobs, verification
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
            _verification=verification,
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
                    item, data, destination, active_blobs, verification
                )
                for name, item in record.items()
            }
        )
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
    The envelope includes path provenance: snapshotting a restored result again
    does not preserve envelope identity or the original restored filenames.
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


class _RestoreVerification:
    """Hold inode witnesses until the complete (including nested) restore ends."""

    def __init__(self, resources: ExitStack) -> None:
        self.resources = resources
        self.directories: dict[Path, int] = {}
        self.files: list[tuple[Path, int, bytes]] = []

    @staticmethod
    def _inode(state: os.stat_result) -> tuple[int, int]:
        return state.st_dev, state.st_ino

    def remember_directory(self, path: Path, descriptor: int) -> None:
        previous = self.directories.get(path)
        if previous is not None:
            if self._inode(os.fstat(previous)) != self._inode(os.fstat(descriptor)):
                raise OSError(f"restore directory changed during restoration: {path}")
            return
        retained = os.dup(descriptor)
        self.resources.callback(os.close, retained)
        self.directories[path] = retained

    def remember_file(self, path: Path, descriptor: int, content: bytes) -> None:
        # Adopt the descriptor returned by _write_exact; keeping it open also
        # prevents inode reuse from making a replacement look like the original.
        self.resources.callback(os.close, descriptor)
        self.files.append((path.absolute(), descriptor, content))

    def _check_binding(self, path: Path, descriptor: int, *, directory: bool) -> None:
        try:
            if path.parent == path:
                current = os.stat(path, follow_symlinks=False)
            else:
                current = os.stat(
                    path.name,
                    dir_fd=self.directories[path.parent],
                    follow_symlinks=False,
                )
        except OSError as exc:
            raise OSError(f"restore path changed during restoration: {path}") from exc
        is_expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if (
            not is_expected_type(current.st_mode)
            or self._inode(current) != self._inode(os.fstat(descriptor))
        ):
            raise OSError(f"restore path changed during restoration: {path}")

    def verify(self) -> None:
        for path, descriptor, content in self.files:
            self._check_binding(path, descriptor, directory=False)
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                handle.seek(0)
                if handle.read() != content:
                    raise OSError(f"restored file content changed: {path}")
        # Check names again after reading, then check the pinned directory chain
        # from leaves to root. None of these lookups follows a symlink.
        for path, descriptor, _ in self.files:
            self._check_binding(path, descriptor, directory=False)
        for path, descriptor in reversed(tuple(self.directories.items())):
            self._check_binding(path, descriptor, directory=True)


def _validate_restored_name(name: Any) -> None:
    # Snapshot-generated names have always used this portable basename alphabet.
    if (
        not isinstance(name, str)
        or name in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
    ):
        raise ValueError(f"invalid native structure restored name: {name!r}")


def _open_directory_component(name: str, parent: int) -> int:
    try:
        return os.open(
            name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise NotADirectoryError(
                f"restore directory must be a directory, not a symlink: {name}"
            ) from exc
        raise


@contextmanager
def _restore_directory(destination: Path, verification: _RestoreVerification):
    """Pin each directory without following links, including existing parents."""
    if not all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW")) or any(
        operation not in os.supports_dir_fd
        for operation in (os.open, os.mkdir, os.stat)
    ) or os.stat not in os.supports_follow_symlinks:
        raise UnsupportedNativeScoresState(
            "safe native score restore requires directory-relative no-follow I/O"
        )
    absolute = destination.absolute()
    current = Path(absolute.anchor)
    descriptor = os.open(current, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        verification.remember_directory(current, descriptor)
        for component in absolute.parts[1:]:
            try:
                child = _open_directory_component(component, descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = _open_directory_component(component, descriptor)
            os.close(descriptor)
            descriptor = child
            current = current / component
            verification.remember_directory(current, descriptor)
        yield descriptor
    finally:
        os.close(descriptor)


def _write_exact(path: Path, content: bytes, directory_fd: int) -> int:
    """Return an owned descriptor for the verified file; never unlink on failure."""
    try:
        descriptor = os.open(
            path.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        created = True
    except FileExistsError:
        # Compare through a pinned file descriptor, never through Path APIs
        # that could follow a link installed between the check and the read.
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise FileExistsError(
                    f"restore destination must not be a symlink: {path}"
                ) from exc
            raise
        created = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FileExistsError(
                f"restore destination already contains different data: {path}"
            )
        with os.fdopen(descriptor, "r+b" if created else "rb", closefd=False) as handle:
            if created:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            elif handle.read() != content:
                raise FileExistsError(
                    f"restore destination already contains different data: {path}"
                )
        return os.dup(descriptor)
    finally:
        # Even a stat-then-unlink cleanup could delete somebody else's new name.
        # Leave partial files for the directory owner rather than risking that.
        os.close(descriptor)


def _restore_structures(
    envelope: NativeScoresEnvelope,
    data: DataStore,
    destination: Path,
    verification: _RestoreVerification,
) -> tuple[Path, ...]:
    if envelope.schema != _SCHEMA:
        raise ValueError(f"unsupported native score envelope schema {envelope.schema!r}")
    names: set[str] = set()
    for structure in envelope.structures:
        _validate_restored_name(structure.restored_name)
        if structure.restored_name in names:
            raise ValueError("duplicate native structure restored name")
        names.add(structure.restored_name)
    targets: list[Path] = []
    with _restore_directory(destination, verification) as directory_fd:
        for structure in envelope.structures:
            content = data.get(structure.content, NATIVE_STRUCTURE_BYTES_CODEC)
            target = destination / structure.restored_name
            descriptor = _write_exact(target, content, directory_fd)
            verification.remember_file(target, descriptor, content)
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
    verification: _RestoreVerification,
) -> Any:
    envelope = data.get(blob, NATIVE_SCORES_ENVELOPE_CODEC)
    if not isinstance(envelope, NativeScoresEnvelope):
        raise TypeError("native score envelope decoded to an unexpected type")

    from astrolabe.evaluators.utils import StandardScores

    expected_fields = tuple(field.name for field in fields(StandardScores))
    extensions = _validate_field_layout(envelope, expected_fields)
    targets = _restore_structures(envelope, data, destination, verification)
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
                verification,
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
    _verification: _RestoreVerification | None = None,
) -> Any:
    """Restore native score semantics and exact selected structure bytes.

    Callers must own/control destination directories and exclude concurrent edits
    to files or ancestor entries; this is not arbitrary-concurrency safety.
    Inode/content checks run at the end of the outer restore, including nested
    scores, but returned Paths cannot prevent subsequent external replacement.
    Failures retain partial files/directories for owner-controlled cleanup.
    Scalar NaNs preserve semantics, not their sign or payload bits.
    """

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
        with ExitStack() as resources:
            verification = (
                _RestoreVerification(resources) if _verification is None else _verification
            )
            restored = _restore_scores(
                blob, data, Path(destination), active_blobs, verification
            )
            if _verification is None:
                verification.verify()
            return restored
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
