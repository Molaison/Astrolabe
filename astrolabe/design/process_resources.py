"""Process-local resource handles for non-durable native objects.

Retained objects are live process resources, not scientific snapshots. A
handle keeps the original Python identity only while its owner permits access;
it does not make mutable scores or poses durable or historically immutable.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
import os
import threading
import uuid
from typing import Any

from .store import DataStore, DataclassCodec
from .types import Blob, Key


@dataclass(frozen=True)
class ProcessReference:
    session: str
    slot: int
    label: str


@dataclass(frozen=True)
class _ProcessResourceEntry:
    value: Any
    label: str
    candidate: Key | None
    revision: int
    expired: bool = False


class ProcessResourceCodec:
    persistent = False

    def __init__(self, owner: ProcessResources) -> None:
        self._owner = owner
        self.identity = f"astrolabe.process-resource/v1/{owner.nonce}"
        self._codec = DataclassCodec(ProcessReference, self.identity)

    def encode(self, value: ProcessReference) -> bytes:
        with self._owner._guard():
            self._owner._entry_for_reference(value)
            return self._codec.encode(value)

    def decode(self, data: bytes) -> ProcessReference:
        with self._owner._guard():
            self._owner._require_alive()
            reference = self._codec.decode(data)
            self._owner._entry_for_reference(reference)
            return reference

    def references(self, value: ProcessReference) -> tuple[Blob, ...]:
        with self._owner._guard():
            self._owner._require_alive()
            self._owner._validate_reference(value)
            return ()

    def compatibility_key(self) -> tuple[str, str]:
        self._owner._require_process()
        return (type(self).__name__, self.identity)

    def validate_scores(
        self,
        blob: Blob,
        scores: Any,
        candidate: Key,
    ) -> tuple[Blob, int]:
        return self._owner.validate_scores(blob, scores, candidate)


class ProcessResources:
    """Own exact in-process objects and expose only non-durable handles.

    A handle references mutable live state and is not a scientific snapshot.
    Native score handles can expire before a state-mutating scientific call;
    their metadata remains decodable while access to the raw score object is
    rejected. Expiry and close cannot revoke a raw reference already returned
    to a caller, nor synchronize mutations of the native object itself.

    One reentrant lock serializes this owner's registration, lookup, expiry,
    close and codec callbacks, including calls through DataStore. It does not
    make unrelated access to the DataStore thread-safe. A DataStore callback
    may reenter on the same thread; it must not wait for another thread to
    access this owner while the callback holds the lock. In particular,
    callbacks must not create cross-owner lock cycles: a thread holding owner
    A must not wait for B while a thread holding B waits for A. Reentrancy is
    local to one thread and owner; this is not general deadlock prevention.

    Handle metadata uses exact Key, str and int types (including Key.digest)
    so equality and hashing cannot invoke user-defined comparison methods.

    Owners inherited across fork are invalid, even when their lock was free.
    Child processes must construct new owners. Non-persistence is enforced
    through this codec and declared Blob reference closures; it cannot stop
    an external codec disguising reference metadata as ordinary bytes.
    """

    def __init__(self, data: DataStore) -> None:
        if not isinstance(data, DataStore):
            raise TypeError("data must be a design DataStore")
        if data.root is not None:
            raise ValueError(
                "ProcessResources requires an in-memory DataStore"
            )
        self.data = data
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self.nonce = uuid.uuid4().hex
        self._next_slot = 0
        self._entries: dict[int, _ProcessResourceEntry] = {}
        self._alive = True
        self._codec = ProcessResourceCodec(self)
        with self._guard():
            data.register(self._codec)

    @property
    def codec(self) -> ProcessResourceCodec:
        self._require_process()
        return self._codec

    def _require_process(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError(
                "process resource owner belongs to another process; "
                "inherited owners are invalid after fork"
            )

    @contextmanager
    def _guard(self) -> Iterator[None]:
        # A fork can inherit an RLock held by a vanished thread. Reject the
        # inherited owner before touching that lock rather than resetting it
        # and accidentally making the parent's handles usable in the child.
        self._require_process()
        with self._lock:
            self._require_process()
            yield

    def _require_alive(self) -> None:
        self._require_process()
        if not self._alive:
            raise RuntimeError("process resource owner is closed")

    def _entry_for_reference(
        self, reference: ProcessReference
    ) -> _ProcessResourceEntry:
        with self._guard():
            self._require_alive()
            self._validate_reference(reference)
            if type(self.nonce) is not str:
                raise TypeError("owner session must be a plain string")
            if reference.session != self.nonce:
                raise ValueError("process resource reference belongs to another session")
            entry = self._entries.get(reference.slot)
            if entry is None:
                raise KeyError("process resource slot is not alive")
            self._validate_label(entry.label)
            self._validate_optional_metadata(
                candidate=entry.candidate, revision=entry.revision,
            )
            if reference.label != entry.label:
                raise ValueError("process resource reference label does not match")
            return entry

    @staticmethod
    def _validate_label(label: str) -> None:
        if type(label) is not str:
            raise TypeError("label must be a plain string")
        if not label:
            raise ValueError("label must be a non-empty string")

    @staticmethod
    def _validate_reference(reference: ProcessReference) -> None:
        if type(reference) is not ProcessReference:
            raise TypeError("expected ProcessReference")
        if type(reference.session) is not str:
            raise TypeError("reference session must be a plain string")
        if type(reference.slot) is not int:
            raise TypeError("reference slot must be a plain integer")
        ProcessResources._validate_label(reference.label)

    @staticmethod
    def _validate_blob(blob: Blob) -> None:
        if type(blob) is not Blob:
            raise TypeError("blob must be a Blob")
        if type(blob.codec) is not str:
            raise TypeError("blob codec must be a plain string")
        if type(blob.identity) is not Key or type(blob.identity.digest) is not str:
            raise TypeError("blob identity must be a Key with a plain string digest")

    @staticmethod
    def _validate_optional_metadata(
        *,
        candidate: Key | None,
        revision: int | None,
        allow_no_revision: bool = False,
    ) -> None:
        if candidate is not None:
            if type(candidate) is not Key:
                raise TypeError("candidate must be a Key or None; subclasses are not allowed")
            if type(candidate.digest) is not str:
                raise TypeError("candidate Key.digest must be a plain string")
        if not (revision is None and allow_no_revision) and type(revision) is not int:
            raise TypeError("revision must be a plain integer")

    def _require_current(
        self, blob: Blob, reference: ProcessReference, entry: _ProcessResourceEntry,
    ) -> None:
        """Recheck the live session and exact entry under the owner's lock."""

        self._require_alive()
        self._validate_blob(blob)
        if self.data._codecs.get(blob.codec) is not self._codec:
            raise ValueError("process resource blob is not owned by this exact session codec")
        current = self._entry_for_reference(reference)
        if current.expired:
            raise RuntimeError(f"process resource {current.label!r} has expired")
        if current is not entry:
            raise RuntimeError("process resource entry is no longer current")

    def retain(
        self,
        value: Any,
        *,
        label: str,
        candidate: Key | None = None,
        revision: int = 0,
    ) -> Blob:
        with self._guard():
            self._require_alive()
            self._validate_label(label)
            self._validate_optional_metadata(
                candidate=candidate,
                revision=revision,
            )
            slot = self._next_slot
            self._next_slot += 1
            entry = _ProcessResourceEntry(
                value=value,
                label=label,
                candidate=candidate,
                revision=revision,
            )
            self._entries[slot] = entry
            reference = ProcessReference(
                session=self.nonce,
                slot=slot,
                label=label,
            )
            try:
                blob = self.data.put(reference, self._codec)
                # The lock permits same-thread callbacks. Such a callback
                # may have closed the owner or expired this newly added score.
                self._require_current(blob, reference, entry)
                return blob
            except BaseException:
                self._entries.pop(slot, None)
                raise

    def resolve(
        self,
        blob: Blob,
        *,
        candidate: Key | None = None,
        revision: int | None = None,
    ) -> Any:
        with self._guard():
            reference, entry = self._resolve_entry(blob, candidate=candidate, revision=revision)
            self._require_current(blob, reference, entry)
            return entry.value

    def _resolve_entry(
        self,
        blob: Blob,
        *,
        candidate: Key | None = None,
        revision: int | None = None,
    ) -> tuple[ProcessReference, _ProcessResourceEntry]:
        """Resolve and validate once; the caller holds the owner lock."""

        self._require_alive()
        self._validate_blob(blob)
        self._validate_optional_metadata(
            candidate=candidate, revision=revision, allow_no_revision=True,
        )
        registered = self.data._codecs.get(blob.codec)
        if registered is not self._codec:
            raise ValueError(
                "process resource blob is not owned by this exact session codec"
            )
        reference = self.data.get(blob, self._codec)
        entry = self._entry_for_reference(reference)
        if entry.expired:
            raise RuntimeError(f"process resource {entry.label!r} has expired")
        if candidate is not None and entry.candidate != candidate:
            raise ValueError("process resource candidate does not match")
        if revision is not None and entry.revision != revision:
            raise ValueError("process resource revision does not match")
        return reference, entry

    def validate_scores(
        self,
        blob: Blob,
        scores: Any,
        candidate: Key,
        revision: int | None = None,
    ) -> tuple[Blob, int]:
        with self._guard():
            self._require_alive()
            if type(candidate) is not Key:
                raise TypeError("candidate must be a Key")
            reference, entry = self._resolve_entry(blob, candidate=candidate, revision=revision)
            if entry.label != "native-scores":
                raise ValueError(
                    "validate_scores requires a native-scores resource"
                )
            if entry.value is not scores:
                raise ValueError(
                    "process resource raw object is not the supplied scores"
                )
            self._require_current(blob, reference, entry)
            return blob, entry.revision

    def expire_scores(self) -> None:
        """Invalidate raw native-score access without changing its handle."""

        with self._guard():
            self._require_alive()
            for slot, entry in tuple(self._entries.items()):
                if entry.label == "native-scores" and not entry.expired:
                    self._entries[slot] = replace(entry, expired=True)

    def close(self) -> None:
        with self._guard():
            self._alive = False
            self._entries.clear()


__all__ = [
    "ProcessReference",
    "ProcessResourceCodec",
    "ProcessResources",
]
