"""Process-scoped native initialization; never retry a possibly partial init."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shlex
import threading
from typing import Any


class PyRosettaInitializationError(RuntimeError):
    """The native process cannot satisfy the requested initialization contract."""


@dataclass(frozen=True)
class PyRosettaInitReceipt:
    state: str
    options: tuple[str, ...] | None
    params: tuple[tuple[str, str], ...]
    version: str | None
    database: str | None
    # No Python/native RNG snapshots or new draws: retain the native init contract.
    rng_contract: str = "native init options; no additional RNG draws"


def pyrosetta_was_initialized(pyrosetta_module: Any) -> bool:
    """Return whether PyRosetta has already been initialized."""
    is_initialized = getattr(pyrosetta_module, "is_initialized", None)
    if callable(is_initialized):
        return bool(is_initialized())
    return bool(pyrosetta_module.rosetta.basic.was_init_called())


def _option_paths(tokens: tuple[str, ...], flags: set[str]) -> tuple[Path, ...]:
    paths = []
    collecting = False
    for token in tokens:
        if token.startswith("-"):
            collecting = token in flags
        elif collecting:
            paths.append(Path(token).expanduser().resolve())
    return tuple(paths)


def _params_identity(paths) -> tuple[tuple[str, str], ...]:
    return tuple(sorted({(str(Path(path).expanduser().resolve()),
                          hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest())
                         for path in paths}))


def _native_version(module: Any) -> str | None:
    version = getattr(module, "version", None)
    if callable(version):
        return str(version())
    value = getattr(module, "__version__", None)
    return str(value) if value is not None else None


def _database_identity(module: Any, tokens: tuple[str, ...]) -> str | None:
    explicit = _option_paths(tokens, {"-database", "-in:path:database"})
    if explicit:
        return str(explicit[-1])
    # PyRosetta's own resolver supplies the same database default used by init.
    resolver = getattr(module, "_rosetta_database_from_env", None)
    value = resolver() if callable(resolver) else None
    return str(Path(value).expanduser().resolve()) if value else None


class PyRosettaInitializationOwner:
    """One immutable receipt per process; external initialization is unattested."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.state = "NEW"
        self.receipt: PyRosettaInitReceipt | None = None
        self.failure: BaseException | None = None

    def _verify_initialized(self, module: Any) -> None:
        try:
            if not pyrosetta_was_initialized(module):
                raise PyRosettaInitializationError(
                    "PyRosetta native runtime is not initialized; use a fresh process"
                )
        except BaseException as exc:
            self.state = "POISONED"
            self.failure = exc
            raise

    def ensure(
        self, module: Any, options: str, *, explicit: bool = False,
        params=(), require_attested: bool = False,
    ) -> PyRosettaInitReceipt:
        with self._lock:
            if self.state in {"STARTING", "POISONED"}:
                raise PyRosettaInitializationError(
                    f"PyRosetta initialization is {self.state}; use a fresh process"
                ) from self.failure
            if self.receipt is not None:
                self._verify_initialized(module)
            elif pyrosetta_was_initialized(module):
                self.state = "INITIALIZED_UNATTESTED"
                self.receipt = PyRosettaInitReceipt(self.state, None, (),
                                                    _native_version(module), None)
            if self.receipt is not None and self.receipt.options is None:
                if require_attested:
                    raise PyRosettaInitializationError(
                        "externally initialized PyRosetta is INITIALIZED_UNATTESTED; "
                        "strict recovery cannot verify its configuration"
                    )
                # Legacy compatibility is not proof of explicit requirements.
                return self.receipt

            tokens = tuple(shlex.split(options))
            required_params = _params_identity(params)
            if self.receipt is not None:
                receipt = self.receipt
                if explicit and tokens != receipt.options:
                    raise PyRosettaInitializationError("conflicting explicit PyRosetta init options")
                current = _params_identity(path for path, _ in receipt.params)
                if current != receipt.params or not set(required_params).issubset(receipt.params):
                    raise PyRosettaInitializationError("conflicting PyRosetta params identity")
                if _native_version(module) != receipt.version:
                    raise PyRosettaInitializationError("conflicting PyRosetta native version")
                if _database_identity(module, receipt.options) != receipt.database:
                    raise PyRosettaInitializationError("conflicting PyRosetta database identity")
            else:
                # Failures here precede native side effects and permit corrected input.
                identities = _params_identity(_option_paths(tokens, {"-extra_res_fa", "-extra_res_cen"}))
                if not set(required_params).issubset(identities):
                    raise PyRosettaInitializationError("required params absent from PyRosetta init options")
                self.state = "STARTING"
                try:
                    module.init(options)
                    self._verify_initialized(module)
                    receipt = PyRosettaInitReceipt(
                        "READY", tokens, identities, _native_version(module),
                        _database_identity(module, tokens),
                    )
                except BaseException as exc:
                    self.state = "POISONED"
                    self.failure = exc
                    raise
                self.receipt = receipt
                self.state = "READY"
            if require_attested and (receipt.version is None or receipt.database is None):
                raise PyRosettaInitializationError("native version/database identity unavailable for strict recovery")
            return receipt


# Runtime/global_config reset deliberately cannot replace this native owner.
_OWNER = PyRosettaInitializationOwner()


def ensure_pyrosetta_initialized(module: Any, options: str, **requirements) -> PyRosettaInitReceipt:
    return _OWNER.ensure(module, options, **requirements)
