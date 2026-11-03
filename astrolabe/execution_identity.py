"""Read-only scientific request identity at the existing trajectory boundary.

This is not another trajectory state store. The sole acceptance writer remains
Decisions/stage_transaction; an execution context carries their observed state
and is discarded after the blocking operation returns.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

_CURRENT = ContextVar("astrolabe_scientific_execution", default=None)
_TRANSACTION = ContextVar("astrolabe_stage_execution", default=None)


def current_execution_identity():
    value = _CURRENT.get()
    return None if value is None else json.loads(json.dumps(value, allow_nan=False))


def _file_fingerprint(path):
    """Read actual contents; metadata alone is not a scientific dependency."""
    path = Path(path).expanduser().absolute()
    resolved = path.resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError("scientific input file changed during snapshot: " + str(path))
    return {"resolved": str(resolved), "sha256": digest.hexdigest(), "bytes": after.st_size}


def input_digest(value, *, file_dependencies=None):
    """Hash supported immutable transport data without consuming any RNG.

    CPU tensors preserve exact dtype/bits. CUDA tensors are deliberately not
    copied or synchronized in a GPU-hidden owner. Unsupported live scientific
    objects must be converted to a proper DTO by their adapter.
    """
    digest = hashlib.sha256()
    stack = set()

    def block(kind, data=b""):
        digest.update(kind.encode() + b":" + len(data).to_bytes(8, "big") + data)

    def file_value(item, hint, explicit=False):
        text = str(item)
        # Never interpret arbitrary sequences/MSA contents as filenames.
        if not text or len(text) > 4096 or "\n" in text or "\x00" in text:
            return False
        name = hint.lower()
        output = any(part in name for part in ("output", "out_dir", "work_dir", "dump_dir", "save_dir", "raw_model_results"))
        relevant = explicit or bool(re.search(r"(?:^|_)(?:path|paths|file|files|filename|pdb|pdbs|cif|mmcif|msa|msas|template|templates)(?:$|_)", name))
        if not relevant or output:
            return False
        path = Path(item).expanduser().absolute()
        try:
            if not path.is_file():
                return False
        except OSError:
            return False
        fingerprint = _file_fingerprint(path)
        block("input_file", json.dumps(fingerprint, sort_keys=True).encode())
        if file_dependencies is not None:
            file_dependencies[str(path)] = fingerprint
        return True

    def feed(item, hint=""):
        if item is None or type(item) in (str, int, float, bool):
            block(type(item).__name__, json.dumps(item, allow_nan=False).encode())
            if isinstance(item, str):
                file_value(item, hint)
            return
        if isinstance(item, Path):
            block("path", str(item).encode())
            file_value(item, hint, explicit=True)
            return
        if isinstance(item, Enum):
            block("enum", type(item).__qualname__.encode())
            feed(item.value)
            return
        if isinstance(item, (bytes, bytearray)):
            block("bytes", bytes(item))
            return
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(item, torch.Tensor):
            if item.device.type != "cpu":
                raise ValueError("scientific request identity requires CPU transport tensors")
            tensor = item.detach().contiguous()
            block("tensor", str(tensor.dtype).encode())
            feed(tuple(tensor.shape))
            if tensor.numel():
                block("bits", tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            return
        numpy = sys.modules.get("numpy")
        if numpy is not None and isinstance(item, numpy.ndarray):
            block("ndarray", item.dtype.str.encode())
            feed(tuple(item.shape))
            if item.dtype.hasobject:
                feed(item.tolist())
            else:
                block("bits", item.tobytes(order="C"))
            return
        if numpy is not None and isinstance(item, numpy.generic):
            block("numpy_scalar", item.dtype.str.encode())
            block("bits", item.tobytes())
            return
        ident = id(item)
        if ident in stack:
            raise ValueError("cyclic object is not a scientific transport snapshot")
        stack.add(ident)
        try:
            if is_dataclass(item) and not isinstance(item, type):
                block("dataclass", type(item).__qualname__.encode())
                for field in fields(item):
                    feed(field.name)
                    feed(getattr(item, field.name), field.name)
            elif isinstance(item, Mapping):
                block("mapping", str(len(item)).encode())
                # Transport maps have string/integer keys. Preserve their types.
                for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                    feed(key)
                    feed(item[key], str(key))
            elif isinstance(item, (tuple, list)):
                block(type(item).__name__, str(len(item)).encode())
                for child in item:
                    feed(child, hint)
            else:
                raise TypeError(f"unsupported live request object: {type(item).__module__}.{type(item).__qualname__}")
        finally:
            stack.remove(ident)

    feed(value)
    return digest.hexdigest()


def _constant(name, default=None):
    from astrolabe.core import global_config
    try:
        value = global_config.get_const(name)
        return default if value is None else value
    except RuntimeError:
        return default


def _owner_snapshot():
    from astrolabe.core import global_config
    decisions = _constant("decisions")
    revision = getattr(decisions, "last_accepted", None)
    assessments = getattr(decisions, "evaluation_results", {})
    try:
        accepted = assessments[revision] if revision is not None else None
    except (IndexError, KeyError, TypeError):
        accepted = None
    manager = _constant("stage_manager")
    gate_context = getattr(manager, "transition_consistency_gate_context", None)
    protomers = _constant("protomers", {})
    sequences = {str(name): {"sequence": getattr(protomer, "sequence", None),
                             "working": getattr(protomer, "mutated_sequence", None),
                             "components": list(getattr(protomer, "components", []))}
                 for name, protomer in protomers.items()}
    segments = _constant("segments", {})
    aliases = {}
    for name, segment in sorted(segments.items()):
        aliases.setdefault(id(segment), []).append(name)
    mapping = sorted(aliases.values())
    return {"trajectory": str(_constant("out_dir", "unbound")),
            "step": global_config.get_variable("step"),
            "accepted_revision": revision,
            "accepted_assessment": getattr(accepted, "assessment_id", None),
            "committed_stage": getattr(manager, "current_stage", None),
            "active_gate": None if gate_context is None else {
                "source_stage": getattr(gate_context, "source_stage", None),
                "target_stage": getattr(gate_context, "target_stage", None),
                "completed_attempts": getattr(manager, "transition_consistency_gate_attempts", 0)},
            "working_sequence_digest": input_digest(sequences),
            "shared_mapping_digest": input_digest(mapping)}


@contextmanager
def stage_transaction_identity(plan, stage_manager):
    """Carry the existing plan before gate result metadata has been written.

    This context owns no stage state and performs no commit. In particular, the
    seed gate's target evaluation precedes begin_seed_transition_gate_attempt.
    """
    if not os.environ.get("ASTROLABE_EXECUTION_ENDPOINT"):
        yield
        return
    metadata = {
        "source_stage": plan.source_stage_name,
        "target_stage": plan.target_stage_name,
        "attempt": int(getattr(stage_manager, "transition_consistency_gate_attempts", 0)) + 1,
        "source_evaluator": plan.source_evaluator_alias,
        "target_evaluator": plan.target_evaluator_alias,
        "gate_evaluator": plan.gate_evaluator_alias,
    }
    token = _TRANSACTION.set(metadata)
    try:
        yield
    finally:
        _TRANSACTION.reset(token)


@contextmanager
def scientific_operation(operation, alias, *, runtime_config=None, inputs=None, reference=None):
    # Preserve the original direct execution path, including its import/RNG and
    # device behavior. Real isolated trajectories have the broker environment.
    if not os.environ.get("ASTROLABE_EXECUTION_ENDPOINT"):
        yield None
        return
    owner = _owner_snapshot()
    runtime = runtime_config or {}
    gate = runtime.get("transition_consistency_gate_metadata") or {}
    if owner["active_gate"] is not None:
        active = owner["active_gate"]
        gate = {"source_stage": active["source_stage"], "target_stage": active["target_stage"],
                "attempt": int(active["completed_attempts"]) + 1}
    if _TRANSACTION.get() is not None:
        gate = _TRANSACTION.get()
    metadata = {str(key): value for key, value in gate.items()
                if value is None or type(value) in (str, int, float, bool)}
    dependencies = {}
    digest = input_digest(inputs, file_dependencies=dependencies)
    identity = {**owner, "scientific_request_id": uuid.uuid4().hex,
                "operation": str(operation), "alias": str(alias),
                "input_digest": digest, "input_files": dependencies, "stage_transaction": metadata,
                "reference_assessment": getattr(reference, "assessment_id", None)}
    parent = _CURRENT.get()
    if parent is not None:
        identity["parent_scientific_request_id"] = parent["scientific_request_id"]
    token = _CURRENT.set(identity)
    try:
        yield identity
        if _owner_snapshot() != owner:
            raise RuntimeError("stale scientific result: trajectory state changed while request was executing")
        for path, expected in dependencies.items():
            try:
                actual = _file_fingerprint(path)
            except OSError as exc:
                raise RuntimeError("stale scientific result: input file became unavailable") from exc
            if actual != expected:
                raise RuntimeError("stale scientific result: input file changed while request was executing")
    finally:
        _CURRENT.reset(token)
