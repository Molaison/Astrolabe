"""Content-addressed receipts for bounded posthoc refold backend outputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator


RECEIPT_SCHEMA = "astrolabe.posthoc-refold.receipt/v1"
IDENTITY_SCHEMA = "astrolabe.posthoc-refold.scientific-identity/v1"
OUTPUT_CONTRACT_SCHEMA = "astrolabe.posthoc-refold.outputs/v1"
NUMERIC_SOURCE_SCHEMA = "astrolabe.posthoc-refold.numeric-source/v2"

_BACKEND_TYPES = {
    "af3": "alphafold3",
    "protenix": "protenix",
}
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class RefoldReceiptConflict(RuntimeError):
    """A target slot cannot be safely reused or recomputed."""


class RefoldReceiptUnknown(RefoldReceiptConflict):
    """A paid request was issued without a terminal proof."""


class RefoldOutputContractConflict(RefoldReceiptConflict):
    """Completed receipt outputs no longer match their committed contract."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RefoldReceiptConflict(
            f"Refold receipt is unreadable: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise RefoldReceiptConflict(
            f"Refold receipt is not a JSON object: {path}"
        )
    return value


def _plain(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(
                value,
                resolve=True,
                throw_on_missing=True,
            )
    except ImportError:
        pass
    return value


def _file_record(path: Path, *, role: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Refold {role} file is unavailable: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"Refold {role} file changed while hashing: {path}")
    return {
        "role": role,
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": after.st_size,
    }


def _path_from_text(value: str) -> Path | None:
    text = value.strip()
    if not text or "\n" in text:
        return None
    candidates = [text]
    path_part, separator, suffix = text.rpartition(":")
    if separator and len(suffix) == 1 and suffix.isalnum():
        candidates.insert(0, path_part)
    for candidate in candidates:
        expanded = os.path.expandvars(os.path.expanduser(candidate))
        path = Path(expanded)
        if path.is_absolute() and path.is_file():
            return path.resolve()
    return None


def _content_record(value: Any, *, role: str) -> Any:
    value = _plain(value)
    if isinstance(value, Path):
        path = value.expanduser().resolve()
        if path.is_file():
            return _file_record(path, role=role)
        return {"$path": str(path), "$exists": path.exists()}
    if isinstance(value, str):
        path = _path_from_text(value)
        if path is not None:
            return _file_record(path, role=role)
        return {
            "$string_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "$length": len(value),
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        return {
            "$bytes_sha256": hashlib.sha256(data).hexdigest(),
            "$length": len(data),
        }
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            return {"$float": repr(value)}
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _content_record(item, role=f"{role}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            _content_record(item, role=f"{role}.{index:06d}")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"Refold identity has unsupported {type(value).__module__}."
        f"{type(value).__qualname__} at {role}"
    )


def _backend_type(backend: str) -> tuple[str, str]:
    canonical = str(backend).strip().lower()
    if canonical == "alphafold3":
        canonical = "af3"
    try:
        return canonical, _BACKEND_TYPES[canonical]
    except KeyError as exc:
        raise ValueError(f"Unsupported refold receipt backend: {backend!r}") from exc


def selected_asset_record(
    *,
    backend: str,
    selected_config: Any,
) -> dict[str, Any]:
    """Reuse the configured-refinement selected-component asset hasher."""

    canonical, type_name = _backend_type(backend)
    from astrolabe.design.native_registry import _asset_identity_record

    effective = {
        "evaluator_terms": {
            canonical: {
                "_type": type_name,
            }
        },
        type_name: _plain(selected_config),
    }
    record = _asset_identity_record(
        family="evaluator",
        type_name=type_name,
        alias=canonical,
        terms_field="evaluator_terms",
        effective_config=effective,
    )
    if record.get("complete") is not True or not record.get("files"):
        raise RefoldReceiptConflict(
            f"Selected {canonical} assets cannot be content-verified"
        )
    return record


def numeric_source_record(
    backend: str,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Content-identify the complete own package and the selected backend."""

    canonical, _type_name = _backend_type(backend)
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    from astrolabe.design.portable_native_builder import content_manifest

    own_files = sorted(
        path.resolve()
        for path in (root / "astrolabe").rglob("*.py")
        if path.is_file()
    )
    run_module = (root / "run.py").resolve()
    if not own_files or not run_module.is_file():
        raise RefoldReceiptConflict(
            f"Own Python source closure is incomplete under {root}"
        )
    own_mapping = {
        path.relative_to(root).as_posix(): path
        for path in (*own_files, run_module)
    }
    own_records = content_manifest(
        own_mapping,
        prefix="numeric-source:own-package",
    )

    origin = None
    third_party_files: list[Path] = []
    third_party_records = ()
    if canonical == "af3":
        try:
            spec = importlib.util.find_spec("alphafold3")
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            raise RefoldReceiptConflict(
                "The runtime alphafold3 source package cannot be resolved"
            ) from exc
        locations = tuple(
            Path(item).expanduser().resolve()
            for item in (spec.submodule_search_locations or ())
        ) if spec is not None else ()
        if not locations:
            raise RefoldReceiptConflict(
                "The runtime alphafold3 source package has no directory origin"
            )
        for location in locations:
            if not location.is_dir():
                raise RefoldReceiptConflict(
                    f"Runtime alphafold3 source directory is unavailable: {location}"
                )
            third_party_files.extend(
                path
                for path in location.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            )
        origin = {
            "kind": "python-package",
            "module": "alphafold3",
            "directories": [str(location) for location in locations],
        }
        third_party_mapping = {
            f"runtime-alphafold3/{path.relative_to(location).as_posix()}": path
            for location in locations
            for path in third_party_files
            if path.is_relative_to(location)
        }
    else:
        protenix_root = (root / "vendor/protenix").resolve()
        if not protenix_root.is_dir():
            raise RefoldReceiptConflict(
                f"Selected protenix package is absent: {protenix_root}"
            )
        third_party_files = [
            path.resolve()
            for path in protenix_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        ]
        third_party_mapping = {
            f"vendored-protenix/{path.relative_to(protenix_root).as_posix()}": path
            for path in third_party_files
        }
        origin = {
            "kind": "vendored-python-package",
            "module": "protenix",
            "directories": [str(protenix_root)],
        }
    if not third_party_mapping:
        raise RefoldReceiptConflict(
            f"Selected {canonical} source package contains no files"
        )
    third_party_records = content_manifest(
        third_party_mapping,
        prefix=f"numeric-source:{canonical}",
    )

    records = (*own_records, *third_party_records)
    files = [
        {
            "role": record.role,
            "path": record.path,
            "sha256": record.sha256,
            "size_bytes": record.size,
        }
        for record in records
    ]
    return {
        "schema": NUMERIC_SOURCE_SCHEMA,
        "backend": canonical,
        "origin": origin,
        "scope": {
            "own_package": "all astrolabe/**/*.py plus run.py",
            "selected_third_party": (
                "the actual selected backend package only"
            ),
            "unselected_backends_or_assets": "not resolved or hashed",
        },
        "own_package_file_count": len(own_records),
        "selected_third_party_file_count": len(third_party_records),
        "files": files,
    }


def build_scientific_identity(
    *,
    backend: str,
    item: Mapping[str, Any],
    config_path: Path,
    selected_config: Any,
    compiled_input_context: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    homologies: Mapping[str, Any],
    asset_record: Mapping[str, Any],
    numeric_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete cross-run identity for one backend slot."""

    canonical, type_name = _backend_type(backend)
    config_path = Path(config_path).expanduser().resolve()
    seqs = item.get("seqs")
    types = item.get("types")
    if not isinstance(seqs, Mapping) or not isinstance(types, Mapping):
        raise ValueError("Refold receipt identity requires mapping seqs/types")
    identity = {
        "schema": IDENTITY_SCHEMA,
        "backend": canonical,
        "backend_type": type_name,
        "step": int(item.get("step")),
        "oligo": str(item.get("oligo") or ""),
        "seed": int(item.get("_receipt_seed", 0)),
        "seqs": {str(key): str(value) for key, value in sorted(seqs.items())},
        "types": {str(key): str(value) for key, value in sorted(types.items())},
        "config_file": _file_record(config_path, role="config"),
        "selected_config": _content_record(
            selected_config,
            role="selected-config",
        ),
        "compiled_input_context": _content_record(
            compiled_input_context,
            role="compiled-input-context",
        ),
        "runtime_config": _content_record(
            runtime_config,
            role="runtime-config",
        ),
        "homologies": _content_record(homologies, role="homologies"),
        "assets": dict(asset_record),
        "numeric_source": dict(numeric_source),
    }
    return identity


def identity_sha256(identity: Mapping[str, Any]) -> str:
    return _digest(identity)


def _snapshot(files: Iterable[Path], *, root: Path) -> list[dict[str, Any]]:
    root = root.expanduser().resolve()
    selected: dict[str, Path] = {}
    for raw_path in files:
        path = Path(raw_path).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RefoldReceiptConflict(
                f"Refold output escapes run root: {path}"
            ) from exc
        if path.is_dir():
            candidates = (
                item for item in path.rglob("*") if item.is_file()
            )
        elif path.is_file():
            candidates = (path,)
        else:
            raise RefoldOutputContractConflict(
                f"Refold output is absent: {path}"
            )
        for candidate in candidates:
            relative = str(candidate.resolve().relative_to(root))
            selected[relative] = candidate
    if not selected:
        raise RefoldOutputContractConflict("Refold output contract is empty")
    records = []
    for relative, path in sorted(selected.items()):
        record = _file_record(path, role=f"output:{relative}")
        record["path"] = relative
        records.append(record)
    return records


def output_contract(files: Iterable[Path], *, run_dir: Path) -> dict[str, Any]:
    records = _snapshot(files, root=run_dir)
    return {
        "schema": OUTPUT_CONTRACT_SCHEMA,
        "fingerprint": _digest(records),
        "files": records,
    }


def validate_output_contract(
    contract: Mapping[str, Any],
    *,
    files: Iterable[Path],
    run_dir: Path,
) -> None:
    if contract.get("schema") != OUTPUT_CONTRACT_SCHEMA:
        raise RefoldOutputContractConflict(
            "Refold output contract has another schema"
        )
    actual = output_contract(files, run_dir=run_dir)
    if actual != dict(contract):
        raise RefoldOutputContractConflict(
            f"Committed refold outputs changed under {run_dir}"
        )


@dataclass(frozen=True)
class RefoldReceiptSlot:
    run_dir: Path
    backend: str
    step: int
    oligo: str

    def __post_init__(self) -> None:
        canonical, _type_name = _backend_type(self.backend)
        object.__setattr__(self, "run_dir", Path(self.run_dir).expanduser().resolve())
        object.__setattr__(self, "backend", canonical)
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "oligo", str(self.oligo))

    @property
    def directory(self) -> Path:
        return (
            self.run_dir
            / ".framework"
            / "refold"
            / self.backend
            / f"step_{self.step:03d}_{self.oligo}"
        )

    @property
    def receipt_path(self) -> Path:
        return self.directory / "receipt.json"

    @property
    def lock_path(self) -> Path:
        return self.directory / "receipt.lock"

    @contextmanager
    def lock(self) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        key = str(self.lock_path)
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(key, threading.Lock())
        if not thread_lock.acquire(blocking=False):
            raise RefoldReceiptConflict(
                f"Refold slot is already active: {self.backend} "
                f"step_{self.step:03d}_{self.oligo}"
            )
        descriptor: int | None = None
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_CREAT | os.O_RDWR,
                0o600,
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                holder = ""
                try:
                    holder = os.pread(descriptor, 4096, 0).decode(
                        "utf-8",
                        errors="replace",
                    ).strip()
                except OSError:
                    pass
                suffix = f"; holder={holder}" if holder else ""
                raise RefoldReceiptConflict(
                    f"Refold slot lock is held: {self.lock_path}{suffix}"
                ) from exc
            owner = json.dumps(
                {
                    "pid": os.getpid(),
                    "thread": threading.get_ident(),
                    "started_unix_s": time.time(),
                },
                sort_keys=True,
            ).encode("utf-8")
            os.ftruncate(descriptor, 0)
            os.pwrite(descriptor, owner + b"\n", 0)
            os.fsync(descriptor)
            yield
        finally:
            if descriptor is not None:
                try:
                    try:
                        os.ftruncate(descriptor, 0)
                        os.fsync(descriptor)
                    except OSError:
                        pass
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
            thread_lock.release()

    def inspect(
        self,
        identity: Mapping[str, Any],
        *,
        marker_present: bool,
        output_roots: Iterable[Path],
        allow_legacy: bool = False,
    ) -> str:
        if not self.receipt_path.exists():
            if not marker_present:
                return "absent"
            if allow_legacy:
                return "legacy"
            raise RefoldReceiptConflict(
                "Target output has an AF3/Protenix marker but no committed "
                f"receipt; refusing to infer or overwrite it: {self.receipt_path}"
            )
        receipt = _read_json(self.receipt_path)
        if receipt.get("schema") != RECEIPT_SCHEMA:
            raise RefoldReceiptConflict(
                f"Refold receipt has an unsupported schema: {self.receipt_path}"
            )
        status = receipt.get("status")
        if status == "issued":
            raise RefoldReceiptUnknown(
                "Refold request was issued without a terminal receipt; model "
                f"side effects are unknown: {self.receipt_path}"
            )
        if status != "completed":
            raise RefoldReceiptConflict(
                f"Refold receipt has unknown status {status!r}: {self.receipt_path}"
            )
        try:
            receipt_step = int(receipt.get("step", -1))
        except (TypeError, ValueError) as exc:
            raise RefoldReceiptConflict(
                f"Refold receipt has an invalid step: {self.receipt_path}"
            ) from exc
        if (
            receipt.get("backend") != self.backend
            or receipt_step != self.step
            or str(receipt.get("oligo")) != self.oligo
        ):
            raise RefoldReceiptConflict(
                f"Refold receipt belongs to another slot: {self.receipt_path}"
            )
        expected_identity = identity_sha256(identity)
        if receipt.get("identity_sha256") != expected_identity:
            raise RefoldReceiptConflict(
                "Committed refold receipt does not match the current scientific "
                f"identity; refusing stale reuse or paid recomputation: {self.receipt_path}"
            )
        contract = receipt.get("outputs")
        if not isinstance(contract, Mapping):
            raise RefoldOutputContractConflict(
                f"Committed refold receipt has no output contract: {self.receipt_path}"
            )
        validate_output_contract(
            contract,
            files=output_roots,
            run_dir=self.run_dir,
        )
        return "reusable"

    def issue(
        self,
        identity: Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        if self.receipt_path.exists():
            raise RefoldReceiptConflict(
                f"Refold receipt already exists; refusing to issue again: {self.receipt_path}"
            )
        _atomic_json(
            self.receipt_path,
            {
                "schema": RECEIPT_SCHEMA,
                "status": "issued",
                "backend": self.backend,
                "step": self.step,
                "oligo": self.oligo,
                "origin": str(origin),
                "identity": dict(identity),
                "identity_sha256": identity_sha256(identity),
                "issued_unix_s": time.time(),
                "issued_pid": os.getpid(),
            },
        )

    def complete(
        self,
        *,
        identity: Mapping[str, Any],
        output_roots: Iterable[Path],
    ) -> None:
        receipt = _read_json(self.receipt_path)
        if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != "issued":
            raise RefoldReceiptConflict(
                f"Refold slot is not issued or is already terminal: {self.receipt_path}"
            )
        expected_identity = identity_sha256(identity)
        if receipt.get("identity_sha256") != expected_identity:
            raise RefoldReceiptConflict(
                f"Issued refold receipt changed identity: {self.receipt_path}"
            )
        contract = output_contract(output_roots, run_dir=self.run_dir)
        receipt.update(
            {
                "status": "completed",
                "outputs": contract,
                "completed_unix_s": time.time(),
                "completed_pid": os.getpid(),
            }
        )
        _atomic_json(self.receipt_path, receipt)
