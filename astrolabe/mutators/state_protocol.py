"""Quiescent scientific-state projections, not persistence or model handles.

The durable caller must bind these snapshots to its phase/RNG receipt. Restore
validates into temporary values before publishing any field. External shared
queues and in-flight/unknown requests are deliberately outside this protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


def json_value(value):
    """Detach JSON data and reject non-finite/non-exportable values."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("Invalid scientific state fields")


def natural(value):
    if type(value) is not int or value < 0:
        raise ValueError("Scientific cursor/count must be a nonnegative integer")
    return value


def file_reference(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_reference(ref):
    fields(ref, ("path", "sha256"))
    if not isinstance(ref["path"], str) or file_reference(ref["path"]) != ref:
        raise ValueError("Scientific structure reference changed")
    return ref["path"]


@dataclass(frozen=True)
class ScientificState:
    """Versioned, detached owner projection with integrity and config binding."""

    protocol: str
    version: int
    identity: dict
    payload: dict
    sha256: str

    @classmethod
    def create(cls, protocol, identity, payload):
        record = json_value(dict(protocol=protocol, version=1, identity=identity, payload=payload))
        return cls(**record, sha256=digest(record))

    def to_dict(self):
        return json_value(vars(self))

    @classmethod
    def validate(cls, state, protocol, identity):
        record = state.to_dict() if isinstance(state, cls) else json_value(state)
        fields(record, ("protocol", "version", "identity", "payload", "sha256"))
        checksum = record.pop("sha256")
        if (record["protocol"] != protocol or type(record["version"]) is not int
                or record["version"] != 1 or digest(record["identity"]) != digest(json_value(identity))
                or checksum != digest(record) or not isinstance(record["payload"], dict)):
            raise ValueError("Incompatible or corrupt scientific state")
        return record["payload"]
