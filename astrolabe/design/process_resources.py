"""Process-local resource handles for non-durable native objects.

Retained objects are live process resources, not scientific snapshots. A
handle keeps the original Python identity only while its owner permits access;
it does not make mutable scores or poses durable or historically immutable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
        self._owner._entry_for_reference(value)
        return self._codec.encode(value)

    def decode(self, data: bytes) -> ProcessReference:
        reference = self._codec.decode(data)
        self._owner._entry_for_reference(reference)
        return reference

    def references(self, value: ProcessReference) -> tuple[Blob, ...]:
        if type(value) is not ProcessReference:
            raise TypeError("process resource codec expects ProcessReference")
        return ()

    def compatibility_key(self) -> tuple[str, str]:
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
    rejected.
    """

    def __init__(self, data: DataStore) -> None:
        if not isinstance(data, DataStore):
            raise TypeError("data must be a design DataStore")
        if data.root is not None:
            raise ValueError(
                "ProcessResources requires an in-memory DataStore"
            )
        self.data = data
        self.nonce = uuid.uuid4().hex
        self._next_slot = 0
        self._entries: dict[int, _ProcessResourceEntry] = {}
        self._alive = True
        self._codec = ProcessResourceCodec(self)
        data.register(self._codec)

    @property
    def codec(self) -> ProcessResourceCodec:
        return self._codec

    def _require_alive(self) -> None:
        if not self._alive:
            raise RuntimeError("process resource owner is closed")

    def _entry_for_reference(
        self, reference: ProcessReference
    ) -> _ProcessResourceEntry:
        self._require_alive()
        if type(reference) is not ProcessReference:
            raise TypeError("expected ProcessReference")
        if reference.session != self.nonce:
            raise ValueError("process resource reference belongs to another session")
        entry = self._entries.get(reference.slot)
        if entry is None:
            raise KeyError("process resource slot is not alive")
        if reference.label != entry.label:
            raise ValueError("process resource reference label does not match")
        return entry

    @staticmethod
    def _validate_optional_metadata(
        *,
        candidate: Key | None,
        revision: int,
    ) -> None:
        if candidate is not None and not isinstance(candidate, Key):
            raise TypeError("candidate must be a Key or None")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise TypeError("revision must be an integer")

    def retain(
        self,
        value: Any,
        *,
        label: str,
        candidate: Key | None = None,
        revision: int = 0,
    ) -> Blob:
        self._require_alive()
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string")
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
            return self.data.put(reference, self._codec)
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
        self._require_alive()
        if not isinstance(blob, Blob):
            raise TypeError("blob must be a Blob")
        registered = self.data._codecs.get(blob.codec)
        if registered is not self._codec:
            raise ValueError(
                "process resource blob is not owned by this exact session codec"
            )
        if candidate is not None:
            if not isinstance(candidate, Key):
                raise TypeError("candidate must be a Key or None")
        if revision is not None and (
            isinstance(revision, bool) or not isinstance(revision, int)
        ):
            raise TypeError("revision must be an integer or None")
        reference = self.data.get(blob, self._codec)
        entry = self._entry_for_reference(reference)
        if entry.expired:
            raise RuntimeError(f"process resource {entry.label!r} has expired")
        if candidate is not None and entry.candidate != candidate:
            raise ValueError("process resource candidate does not match")
        if revision is not None and entry.revision != revision:
            raise ValueError("process resource revision does not match")
        return entry.value

    def validate_scores(
        self,
        blob: Blob,
        scores: Any,
        candidate: Key,
        revision: int | None = None,
    ) -> tuple[Blob, int]:
        if not isinstance(candidate, Key):
            raise TypeError("candidate must be a Key")
        value = self.resolve(
            blob,
            candidate=candidate,
            revision=revision,
        )
        registered = self.data._codecs[blob.codec]
        reference = self.data.get(blob, registered)
        entry = self._entry_for_reference(reference)
        if entry.label != "native-scores":
            raise ValueError(
                "validate_scores requires a native-scores resource"
            )
        if value is not scores:
            raise ValueError(
                "process resource raw object is not the supplied scores"
            )
        return blob, entry.revision

    def expire_scores(self) -> None:
        """Invalidate raw native-score access without changing its handle."""

        self._require_alive()
        for slot, entry in tuple(self._entries.items()):
            if entry.label == "native-scores" and not entry.expired:
                self._entries[slot] = replace(entry, expired=True)

    def close(self) -> None:
        self._alive = False
        self._entries.clear()


__all__ = [
    "ProcessReference",
    "ProcessResourceCodec",
    "ProcessResources",
]
