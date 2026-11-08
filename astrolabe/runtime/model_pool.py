"""Explicit, bounded residency inside one persistent scientific worker.

NativeModelClient may acquire a borrowed owner in ``with local_model_pool(...)``
and fall back to its private owner when acquire returns None. The parent must
consult pool_owns_registration() before adding a newly constructed owner to a
model_run_scope. Only this pool shuts down its real owners; allocator/broker
residency remains charged until the persistent process actually exits.

No framework import, CUDA probe, RNG capture, or RNG restoration happens here.
Every request still traverses the real owner's bounded pickle transport.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import importlib
import logging
import os
from pathlib import Path
import sys
import threading


_ACTIVE_POOL: LocalModelPool | None = None
_CUDA_FIELDS = frozenset({"cuda", "cuda_seed", "cuda_visible_devices"})


class LifecycleDependencyError(RuntimeError):
    """A retained dependency cannot reproduce its original CPU RNG effects."""


def import_lifecycle_dependency(name):
    """Retain an import's evidenced CPU RNG effects within an explicit pool.

    Some optional CPU libraries consume global RNG while importing. Repeated
    trajectories must reproduce that effect before acquiring the model. This
    does not relax model reuse's complete initial RNG or CUDA proof.
    """
    pool = get_local_model_pool()
    if pool is None:
        return importlib.import_module(name)
    return pool.import_dependency(name)


def _main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("Local model pooling requires the main thread")


def _same(left, right):
    """Exact captured-state comparison, including absent keys and value types."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(map(_same, left, right))
    return left == right


def _rng_matches(expected, actual, initrecord):
    if _same(expected, actual):
        return True
    # The sole allowed representation transition is a saved lazy state followed
    # by a caller's fully initialized CUDA state. A seed is never a state proof.
    if ("cuda_seed" not in expected or "cuda" in expected
            or "cuda" not in actual or "cuda_seed" in actual):
        return False
    transition = initrecord.get("metadata", {}).get("initialization_transition")
    if not isinstance(transition, Mapping):
        return False
    before = transition.get("before")
    placed = transition.get("cuda_before_place")
    after = transition.get("after")
    if not all(isinstance(state, Mapping) for state in (before, placed, after)):
        return False
    if not _same(before, expected) or not _same(after, initrecord.get("random_state")):
        return False
    cpu = lambda state: {key: value for key, value in state.items()
                         if key not in _CUDA_FIELDS}
    if not _same(cpu(expected), cpu(actual)):
        return False
    if not all("cuda" in state and "cuda_visible_devices" in state
               for state in (actual, placed, after)):
        return False
    seed = expected["cuda_seed"]
    if not all("torch_seed" in state and _same(seed, state["torch_seed"])
               for state in (expected, actual, placed, after)):
        return False
    return (_same(actual["cuda"], placed["cuda"])
            and _same(actual["cuda_visible_devices"], placed["cuda_visible_devices"]))


def _scope():
    # Preserve None versus empty, ordering, and the complete visibility mapping.
    return (os.getpid(), os.environ.get("ASTROLABE_RESOURCE_SOCKET"),
            os.environ.get("CUDA_VISIBLE_DEVICES"))


def get_local_model_pool() -> LocalModelPool | None:
    pool = _ACTIVE_POOL
    if pool is None or pool._pid != os.getpid() or pool._closed:
        return None
    _main_thread()
    return pool


def pool_owns_registration(owner) -> bool:
    """Claim only the real owner registering during this pool's construction.

    Called by model_execution.register_model_owner before run-scope registration.
    Unrelated/nested owners and later registrations are not claimed.
    """
    pool = get_local_model_pool()
    if (pool is None or pool._constructing_spec is None
            or getattr(owner, "spec", None) is not pool._constructing_spec):
        return False
    pool._remember_owner(owner)
    return True


@dataclass
class _Entry:
    owner: object
    identity: str
    backend: str
    factory: object
    owner_kwargs: dict
    initial_state: dict
    profile: object
    initrecord: dict
    hits: int = 0
    borrowers: int = 0


class BorrowedOwner:
    """An attached handle; shutdown releases this handle, never the model."""

    attached = True
    release_requires_process_exit = True

    def __init__(self, pool, entry):
        self._pool = pool
        self._entry = entry
        self._closed = False
        entry.borrowers += 1

    @property
    def pid(self):
        return self._entry.owner.pid

    @property
    def identity(self):
        return self._entry.identity

    @property
    def spec(self):
        return self._entry.owner.spec

    @property
    def temp_dir(self):
        return getattr(self._entry.owner, "temp_dir", None)

    @property
    def log_path(self):
        return getattr(self._entry.owner, "log_path", None)

    @property
    def socket_path(self):
        return getattr(self._entry.owner, "socket_path", None)

    @property
    def initrecord(self):
        return deepcopy(self._entry.initrecord)

    def request(self, request):
        self._pool._check_request()
        if self._closed:
            raise RuntimeError("Borrowed model owner is closed")
        self._pool._in_request = True
        try:
            response = self._entry.owner.request(request)
            if not isinstance(response, Mapping) or response.get("ok") is not True:
                self._pool._quarantined = True
                return response
            if request.get("operation") == "describe":
                # Preserve the real post-init RNG effects across trajectories.
                # The real request above still validates input/output bounds.
                return self.initrecord
            return response
        except BaseException:
            # Includes unknown outcomes: never replay, replace, or reload.
            self._pool._quarantined = True
            raise
        finally:
            self._pool._in_request = False

    def shutdown(self):
        self._pool._check_process()
        if not self._closed:
            self._closed = True
            self._entry.borrowers -= 1

    def quarantine(self):
        """A caller could not validate a possibly executed response."""
        self._pool._check_process()
        self._pool._quarantined = True


class LocalModelPool:
    """A main-thread context with a fixed owner cap and no eviction policy.

    Incompatible requests and exhausted capacity return None. Construction or
    request failures quarantine the entire pool and raise on subsequent use,
    preventing an accidental private-owner fallback from replaying failed work.
    """

    def __init__(self, *, max_models=1, backends=frozenset({"esmfold"})):
        if type(max_models) is not int or max_models < 1:
            raise ValueError("max_models must be a positive integer")
        if isinstance(backends, str):
            raise ValueError("backends must be a collection of backend names")
        self.backends = frozenset(backends)
        if not self.backends or any(not isinstance(item, str) or not item
                                    for item in self.backends):
            raise ValueError("backends must contain nonempty names")
        self.max_models = max_models
        self._pid = os.getpid()
        self._scope = None
        self._entered = False
        self._closed = False
        self._quarantined = False
        self._in_request = False
        self._constructing_spec = None
        self._owners = []
        self._entries = []
        self._real_load_count = 0
        self._dependencies = {}
        self._cleanup_errors = []

    def _check_process(self):
        _main_thread()
        if os.getpid() != self._pid:
            raise RuntimeError("Local model pool belongs to another process")

    def _check_request(self, *, check_scope=True):
        self._check_process()
        if self._closed or not self._entered:
            raise RuntimeError("Local model pool is closed or not entered")
        if self._quarantined:
            raise RuntimeError("Local model pool is quarantined after a failed request or load")
        if self._in_request or self._constructing_spec is not None:
            raise RuntimeError("Local model pool does not allow reentrant model execution")
        if check_scope and _scope() != self._scope:
            raise RuntimeError("Local model pool broker or CUDA visibility scope changed")

    def __enter__(self):
        global _ACTIVE_POOL
        self._check_process()
        if self._entered or self._closed or get_local_model_pool() is not None:
            raise RuntimeError("Local model pool contexts cannot be nested or reentered")
        self._scope = _scope()
        self._entered = True
        _ACTIVE_POOL = self
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
            logging.exception("Failed to close local model pool during exception cleanup")
        return False

    def _remember_owner(self, owner):
        if not any(item is owner for item in self._owners):
            self._owners.append(owner)

    def import_dependency(self, name):
        try:
            self._check_request()
            return self._import_dependency(name)
        except BaseException as error:
            # Import/capture/restore can fail after changing global state.
            # Keep all owners for cleanup, but never continue or record a
            # subsequent cached import as a new successful initialization.
            self._quarantined = True
            if not isinstance(error, Exception) or isinstance(error, LifecycleDependencyError):
                raise
            raise LifecycleDependencyError(str(error)) from error

    def _import_dependency(self, name):
        from astrolabe.model_execution import capture_random_state, restore_random_state

        before = capture_random_state()
        cpu = lambda state: {key: value for key, value in state.items()
                             if key not in _CUDA_FIELDS}
        saved = self._dependencies.get(name)
        if saved is not None:
            if not _same(cpu(before), cpu(saved["before"])):
                raise LifecycleDependencyError(
                    f"Dependency {name} received a different complete CPU RNG state")
            module = importlib.import_module(name)
            if (module is not saved["module"]
                    or not _same(before, capture_random_state())):
                raise LifecycleDependencyError(
                    f"Dependency {name} changed module or RNG during cached import")
            # Preserve the actual complete current CUDA state. The saved first
            # import is only eligible when it demonstrably left CUDA unchanged.
            restore_random_state(dict(before, **cpu(saved["after"])))
            saved["hits"] += 1
            return module
        module = importlib.import_module(name)
        after = capture_random_state()
        if not _same({k: v for k, v in before.items() if k in _CUDA_FIELDS},
                     {k: v for k, v in after.items() if k in _CUDA_FIELDS}):
            raise LifecycleDependencyError(
                f"Dependency {name} changed CUDA; CPU import reuse is unsupported")
        self._dependencies[name] = {
            "module": module, "before": deepcopy(before), "after": deepcopy(after), "hits": 0,
        }
        return module

    def acquire(self, spec, owner_factory, owner_kwargs, identity) -> BorrowedOwner | None:
        self._check_request(check_scope=False)
        # Scope mismatch is an incompatibility, not a request failure.
        if _scope() != self._scope:
            return None
        if (spec.backend not in self.backends or spec.provider is not None
                or owner_kwargs.get("socket_path") is not None
                or Path(spec.python_bin).resolve() != Path(sys.executable).resolve()
                or (spec.cwd and Path(spec.cwd).resolve() != Path.cwd().resolve())
                or any(str(value) != os.environ.get(str(key))
                       for key, value in (spec.env or {}).items())):
            return None
        state = spec.settings.get("_initial_random_state")
        if not isinstance(state, dict) or not state:
            return None
        profile = spec.settings.get("_runtime_profile")
        for entry in self._entries:
            if (entry.identity == identity and entry.backend == spec.backend
                    and entry.factory is owner_factory
                    and _same(entry.owner_kwargs, owner_kwargs)
                    and _same(entry.profile, profile)
                    and _rng_matches(entry.initial_state, state, entry.initrecord)):
                entry.hits += 1
                return BorrowedOwner(self, entry)
        if any(entry.identity == identity for entry in self._entries):
            return None
        if len(self._owners) >= self.max_models:
            return None
        # Snapshot before a factory can mutate the supplied settings/kwargs.
        initial_state = deepcopy(state)
        saved_kwargs = deepcopy(owner_kwargs)
        saved_profile = deepcopy(profile)
        self._constructing_spec = spec
        try:
            owner = owner_factory(spec, **owner_kwargs)
            self._remember_owner(owner)
            self._real_load_count += 1
            description = owner.request({"operation": "describe"})
            if (not isinstance(description, Mapping) or description.get("ok") is not True
                    or not isinstance(description.get("metadata"), Mapping)
                    or not isinstance(description.get("random_state"), Mapping)):
                raise RuntimeError("Real model owner did not return a valid initialization record")
            entry = _Entry(owner, identity, spec.backend, owner_factory, saved_kwargs,
                           initial_state, saved_profile, deepcopy(description))
            self._entries.append(entry)
            return BorrowedOwner(self, entry)
        except BaseException:
            self._quarantined = True
            raise
        finally:
            self._constructing_spec = None

    @property
    def metadata(self):
        return {"pid": self._pid, "max_models": self.max_models,
                "real_load_count": self._real_load_count,
                "hits": sum(entry.hits for entry in self._entries),
                "closed": self._closed, "quarantined": self._quarantined,
                "cleanup_complete": self._closed and not self._cleanup_errors,
                "cleanup_errors": list(self._cleanup_errors),
                "release_requires_process_exit": True,
                "dependency_initializations": [
                    {"module": name, "source": getattr(row["module"], "__file__", None),
                     "hits": row["hits"], "before": deepcopy(row["before"]),
                     "after": deepcopy(row["after"]), "cuda_effect": "unchanged"}
                    for name, row in self._dependencies.items()],
                "models": [{"identity": entry.identity, "pid": entry.owner.pid,
                            "model_id": entry.initrecord["metadata"].get(
                                "runtime_lifecycle", {}).get("model_id"),
                            "hits": entry.hits, "borrowers": entry.borrowers,
                            "initrecord": deepcopy(entry.initrecord)}
                           for entry in self._entries]}

    def close(self):
        global _ACTIVE_POOL
        self._check_process()
        if self._closed:
            return
        if self._in_request or self._constructing_spec is not None:
            raise RuntimeError("Cannot close a local model pool during model execution")
        self._closed = True
        if _ACTIVE_POOL is self:
            _ACTIVE_POOL = None
        first_error = None
        for owner in reversed(self._owners):
            try:
                owner.shutdown()
            except BaseException as error:
                self._quarantined = True
                self._cleanup_errors.append(repr(error))
                if first_error is None:
                    first_error = error
        # No unregister_model: shutdown cannot prove allocator/VRAM release.
        if first_error is not None:
            raise RuntimeError("A real local model owner failed to close") from first_error


def local_model_pool(*, max_models=1, backends=frozenset({"esmfold"})) -> LocalModelPool:
    """Create the explicit residency context (inactive until entered)."""
    return LocalModelPool(max_models=max_models, backends=backends)
