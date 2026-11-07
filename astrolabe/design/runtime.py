"""Durable typed calls, with conservative recovery and no inference replay.

Runtime is a scientific journal, not a device scheduler. An operation may use
the existing model owners/PipelineExecutor underneath its invocation boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import sqlite3

from astrolabe.model_execution import capture_random_state, restore_random_state
from astrolabe.model_daemon import (
    DaemonChildExited,
    DaemonDisconnected,
    RequestOutcomeUnknown,
)
from .operations import INVOKE
from .owner import SCIENTIFIC_OWNER
from .store import ArrayCodec, DataStore, JSONCodec, design_codec
from .types import (
    Attempt, Blob, CallKey, History, Key, Preflight, Reconciliation, RNGReceipt,
    key_for,
)

_SCIENTIFIC_OWNER = SCIENTIFIC_OWNER


class ContractConflict(ValueError):
    """A recorded call/work was reentered with a different scientific input."""


class OutcomeUnknown(RuntimeError):
    def __init__(self, attempt: str):
        self.attempt = attempt
        super().__init__(f"Attempt {attempt} has no verified terminal outcome; automatic replay is blocked")


class RecordedFailure(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass


def exception_chain(error: BaseException):
    """Yield every explicit cause/context and ExceptionGroup member once."""
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(
            cause
            for cause in (getattr(current, "__cause__", None),
                          getattr(current, "__context__", None))
            if cause is not None
        )
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def outcome_is_unknown(error: BaseException) -> bool:
    uncertain_types = (
        OutcomeUnknown,
        RequestOutcomeUnknown,
        ContractConflict,
        BrokenProcessPool,
        DaemonDisconnected,
        DaemonChildExited,
        TimeoutError,
        ConnectionError,
        EOFError,
    )
    return any(
        isinstance(cause, uncertain_types)
        or bool(getattr(cause, "outcome_unknown", False))
        for cause in exception_chain(error)
    )


def _blob_record(blob):
    return {"identity": blob.identity.digest, "codec": blob.codec} if blob else None


def _blob(record):
    return Blob(Key(record["identity"]), record["codec"]) if record else None


class Runtime:
    def __init__(self, operations=(), *, root: Path | str | None = None,
                 data: DataStore | None = None, budgets=None,
                 scientific_contract: Key | None = None, on_checkpoint=None):
        self.root = Path(root).resolve() if root is not None else None
        self.data = data if data is not None else DataStore(self.root / "data" if self.root else None)
        self.data.register(design_codec)
        self._json = JSONCodec()
        self.data.register(self._json)
        self.data.register(ArrayCodec())
        self.operations = {}
        for component in operations:
            alias = component.descriptor.alias
            if alias in self.operations:
                raise ValueError(f"Duplicate component alias {alias}")
            self.operations[alias] = component
            self.data.register(component.descriptor.request_codec)
            self.data.register(component.descriptor.result_codec)
            for codec in getattr(component, "reference_codecs", ()):
                self.data.register(codec)
        self.budgets = dict(budgets or {})
        if any(type(n) is not int or n < 0 for n in self.budgets.values()):
            raise ValueError("Operation budgets must be nonnegative integers")
        self.scientific_contract = scientific_contract or key_for((
            "astrolabe.design.runtime/v1", self.budgets,
            tuple((k, v.descriptor.identity) for k, v in sorted(self.operations.items()))))
        self.history = History(key_for(()), ())
        self._memory = {}
        self._connection = None
        self._lock_file = None
        self._work = None
        self._preflight = None
        self._declared = frozenset()
        self._closed = False
        self.on_checkpoint = on_checkpoint
        try:
            if self.root is not None:
                self.root.mkdir(parents=True, exist_ok=True)
                self._lock_file = (self.root / "owner.lock").open("a+b")
                fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._connection = sqlite3.connect(str(self.root / "journal.sqlite"))
                self._connection.execute("PRAGMA synchronous=FULL")
                # Reuse the rollback journal's allocation instead of unlinking
                # and recreating it for every durable checkpoint. FULL still
                # syncs the commit (zeroed journal header); no shared-memory WAL.
                self._connection.execute("PRAGMA journal_mode=PERSIST")
                self._connection.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT, name TEXT, value TEXT, PRIMARY KEY(kind,name))")
            existing = self._read("meta", "contract")
            contract = {"revision": self.scientific_contract.digest,
                        "operations": {k: v.descriptor.identity.digest for k, v in self.operations.items()},
                        "budgets": self.budgets, "journal_version": 1}
            if existing is not None and existing != contract:
                raise ContractConflict("Runtime scientific contract, components or budgets changed on resume")
            if existing is not None:
                unverified = [
                    alias for alias, operation in self.operations.items()
                    if getattr(operation, "resume_assets_verified", True) is False
                ]
                if unverified:
                    raise ContractConflict(
                        "Cannot resume: selected native component assets cannot be "
                        f"content-verified: {sorted(unverified)!r}"
                    )
            if existing is None:
                self._write("meta", "contract", contract)
        except BaseException:
            self.close()
            raise

    def _read(self, kind, name):
        if self._connection is None:
            return deepcopy(self._memory.get((kind, name)))
        row = self._connection.execute("SELECT value FROM records WHERE kind=? AND name=?", (kind, name)).fetchone()
        return json.loads(row[0]) if row else None

    def _write(self, kind, name, value):
        if self._closed:
            raise RuntimeError("Runtime is closed")
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if self._connection is None:
            self._memory[kind, name] = json.loads(payload)
        else:
            with self._connection:
                self._connection.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)", (kind, name, payload))

    def _records(self, kind):
        if self._connection is None:
            return {name: deepcopy(value) for (k, name), value in self._memory.items() if k == kind}
        return {name: json.loads(value) for name, value in
                self._connection.execute("SELECT name,value FROM records WHERE kind=?", (kind,))}

    def _durable(self, *references):
        if self.root is not None:
            self.data.persist(tuple(ref for ref in references if ref is not None))

    def save(self, name, value, codec=design_codec):
        reference = self.data.put(value, codec)
        self._durable(reference)
        self._write("checkpoint", name, _blob_record(reference))
        if self.on_checkpoint is not None:
            before = self._capture_rng()
            try:
                self.on_checkpoint(name, reference)
                if self._capture_rng() != before:
                    raise ValueError("Checkpoint reporting consumed scientific RNG")
            finally:
                self._restore_rng(before)
        return reference

    def load(self, name, codec=design_codec):
        record = self._read("checkpoint", name)
        return self.data.get(_blob(record), codec) if record else None

    def _capture_rng(self):
        return self.data.put(capture_random_state(), design_codec)

    def _restore_rng(self, reference):
        restore_random_state(self.data.get(reference, design_codec))

    def _save_completion(self, name, value, codec=design_codec):
        result = self.data.put(value, codec)
        self.save(name, {"result": result, "rng": self._capture_rng()})

    def _load_completion(self, name, codec=design_codec):
        envelope = self.load(name)
        if envelope is None:
            return None
        self._restore_rng(envelope["rng"])
        return self.data.get(envelope["result"], codec)

    def peek_completion(self, name, codec=design_codec):
        """Read a completed projection without changing the active RNG stream."""
        envelope = self.load(name)
        return None if envelope is None else self.data.get(envelope["result"], codec)

    def work_entry(self, work_id):
        from .types import Work
        saved = self._read("work", work_id)
        if saved is None:
            raise KeyError(work_id)
        _, entry = self.data.get(_blob(saved["payload"]), design_codec)
        if not isinstance(entry, Work):
            raise ValueError("The requested entry is not search Work")
        return entry

    def _deliver(self, operation, result):
        """Rebind owner metadata after decoding; no model work is allowed here."""
        callback = getattr(operation, "on_result", None)
        if callback is not None:
            before = self._capture_rng()
            try:
                callback(result)
                if self._capture_rng() != before:
                    raise ValueError("Result binding consumed RNG")
            except Exception as error:
                self._restore_rng(before)
                raise ContractConflict("Operation result metadata could not be rebound") from error
        return result

    def preflight(self, requirements):
        requirements = tuple(requirements)
        if self._preflight is not None:
            if self._preflight.requirements != requirements:
                raise ContractConflict("Trial requirements changed after preflight")
            return self._preflight
        for requirement in requirements:
            component = self.operations.get(requirement.component)
            available = component is not None and requirement.capability in component.descriptor.capabilities
            if not available and requirement.enforcement == "required":
                raise ValueError(f"Missing required capability: {requirement.component}: {requirement.capability}")
        self._declared = frozenset(r.component for r in requirements)
        self._preflight = Preflight(key_for((requirements, self.scientific_contract)),
                                    requirements, self.scientific_contract)
        self.save("preflight", self._preflight)
        return self._preflight

    @contextmanager
    def bind(self, work_id: str, history: History, *, entry=None):
        """Reenter deterministic Trial control flow from its immutable entry.

        Scientific globals/RNG currently have one explicit serial owner. Device
        execution can still pipeline *inside* an operation's existing owner.
        """
        if self._preflight is None:
            raise RuntimeError("Preflight is required before work entry")
        if self._work is not None:
            raise RuntimeError("Nested work binding is not supported")
        with _SCIENTIFIC_OWNER:
            entry_key = key_for((work_id, history, entry))
            saved = self._read("work", work_id)
            if saved is None:
                state = self.data.put(capture_random_state(), design_codec)
                payload = self.data.put((history, entry), design_codec)
                self._durable(state, payload)
                saved = {"entry": entry_key.digest, "payload": _blob_record(payload),
                         "rng": _blob_record(state)}
                self._write("work", work_id, saved)
            elif saved["entry"] != entry_key.digest:
                raise ContractConflict(f"Work {work_id!r} entry changed")
            restore_random_state(self.data.get(_blob(saved["rng"]), design_codec))
            previous = self.history
            self.history, self._work = history, work_id
            try:
                yield self
            finally:
                self.history, self._work = previous, None

    def call(self, named_round, operation, request, *, work_id=None):
        if self._work is None:
            raise RuntimeError("Runtime.call requires a bound work entry")
        if not named_round or named_round.startswith("/") or ".." in named_round.split("/"):
            raise ValueError("Calls require a stable relative named_round")
        call_work_id = self._work if work_id is None else work_id
        if not isinstance(call_work_id, str) or not call_work_id:
            raise ValueError("Call work_id must be a non-empty string")
        if call_work_id.startswith("/") or ".." in call_work_id.split("/"):
            raise ValueError("Call work_id must be a stable relative identity")
        descriptor = operation.descriptor
        registered = self.operations.get(descriptor.alias)
        if registered is not operation or descriptor.alias not in self._declared:
            raise ValueError(f"Undeclared operation {descriptor.alias!r}")
        preparation_rng = capture_random_state()
        try:
            prepared = operation.prepare(request)
        except BaseException:
            restore_random_state(preparation_rng)
            raise
        after_preparation = capture_random_state()
        if any(key_for(value) != key_for(after_preparation.get(name))
               for name, value in preparation_rng.items()):
            restore_random_state(preparation_rng)
            raise ValueError("Operation preparation consumed RNG; use a named scientific operation")
        for requirement in prepared.requirements:
            component = self.operations.get(requirement.component)
            if (requirement.component not in self._declared or component is None or
                    requirement.capability not in component.descriptor.capabilities):
                if requirement.enforcement == "required":
                    raise ValueError(f"Request requires undeclared/unsupported capability {requirement}")
        request_ref = self.data.put(prepared.request, descriptor.request_codec)
        if not isinstance(prepared.references, tuple) or any(
                not isinstance(reference, Blob) for reference in prepared.references):
            raise TypeError("Prepared references must be a tuple of immutable Blobs")
        key = CallKey(call_work_id, named_round, descriptor.identity, prepared.inputs.identity)
        slot = key_for((call_work_id, named_round)).digest
        previous = self._read("attempt", slot)
        binding = key_for((key, request_ref, prepared.paid_boundary, prepared.references)).digest
        if previous is not None and previous["binding"] != binding:
            raise ContractConflict(
                f"Inputs or component changed at {call_work_id}/{named_round}"
            )
        if previous is not None:
            if previous["status"] == "completed":
                receipt = self.data.get(_blob(previous["rng"]), design_codec)
                restore_random_state(self.data.get(receipt.after, design_codec))
                return self._deliver(operation, self.data.get(_blob(previous["result"]), descriptor.result_codec))
            if previous["status"] in ("issued", "unknown"):
                if previous["status"] == "issued":
                    previous["status"] = "unknown"
                    self._write("attempt", slot, previous)
                raise OutcomeUnknown(slot)
            if previous["status"] == "failed":
                receipt = self.data.get(_blob(previous["rng"]), design_codec)
                restore_random_state(self.data.get(receipt.after, design_codec))
                raise RecordedFailure(previous.get("error", f"Attempt {slot} failed"))
        else:
            boundary = prepared.paid_boundary
            if boundary in self.budgets:
                used = sum(row["boundary"] == boundary for row in self._records("attempt").values())
                if used >= self.budgets[boundary]:
                    raise BudgetExhausted(f"{boundary}: operation budget exhausted")
            before = self.data.put(capture_random_state(), design_codec)
            key_ref = self.data.put(key, design_codec)
            inputs_ref = self.data.put(prepared.inputs, design_codec)
            self._durable(request_ref, key_ref, inputs_ref, before, *prepared.references)
            previous = dict(binding=binding, component=descriptor.alias, boundary=boundary,
                status="reserved", key=_blob_record(key_ref), request=_blob_record(request_ref),
                references=[_blob_record(reference) for reference in prepared.references],
                inputs=_blob_record(inputs_ref), before=_blob_record(before), result=None, rng=None)
            self._write("attempt", slot, previous)
        # A reserved-but-unissued call may resume; issued work never auto-replays.
        before = _blob(previous["before"])
        restore_random_state(self.data.get(before, design_codec))
        previous["status"] = "issued"
        self._write("attempt", slot, previous)
        returned = False
        try:
            result = operation.invoke(prepared)
            returned = True
            reference = self.data.put(result, descriptor.result_codec)
            after = self.data.put(capture_random_state(), design_codec)
            receipt_ref = self.data.put(RNGReceipt(before, after, self.scientific_contract), design_codec)
            self._durable(reference, receipt_ref)
        except BaseException as error:
            try:
                chain_unknown = outcome_is_unknown(error)
                if returned or not isinstance(error, Exception) or chain_unknown:
                    # Lost/ambiguous and non-ordinary outcomes are never
                    # terminal evidence, even if a broad verifier says so.
                    terminal = False
                    unknown = True
                else:
                    verifier = getattr(operation, "failure_is_terminal", None)
                    try:
                        terminal = (not prepared.paid_boundary if verifier is None else
                                    verifier(error, paid_boundary=prepared.paid_boundary))
                    except Exception:
                        terminal = False
                    unknown = not terminal
                previous.update(status="unknown" if unknown else "failed",
                                error=f"{type(error).__name__}: {error}")
                after = self.data.put(capture_random_state(), design_codec)
                receipt_ref = self.data.put(RNGReceipt(before, after, self.scientific_contract), design_codec)
                self._durable(receipt_ref)
                previous["rng"] = _blob_record(receipt_ref)
                self._write("attempt", slot, previous)
            except Exception as recording_error:
                # The durable issued record remains authoritative if terminal
                # recording itself fails. Never certify the request as failed.
                raise OutcomeUnknown(slot) from recording_error
            if unknown and isinstance(error, Exception):
                raise OutcomeUnknown(slot) from error
            raise
        previous.update(status="completed", result=_blob_record(reference), rng=_blob_record(receipt_ref))
        try:
            self._write("attempt", slot, previous)
        except Exception as commit_error:
            raise OutcomeUnknown(slot) from commit_error
        # The first consumer receives the same frozen content as a recovered
        # consumer, never an alias into a model/adapter's mutable result cache.
        return self._deliver(operation, self.data.get(reference, descriptor.result_codec))

    def attempts(self, work_id=None):
        results = []
        for identity, row in self._records("attempt").items():
            key = self.data.get(_blob(row["key"]), design_codec)
            if work_id is None or key.work_id == work_id:
                results.append(Attempt(identity, key, "unknown" if row["status"] == "issued" else row["status"],
                                       _blob(row["result"]),
                                       self.data.get(_blob(row["rng"]), design_codec) if row["rng"] else None))
        return tuple(results)

    def reconcile(self, attempt):
        row = self._read("attempt", attempt)
        if row is None:
            raise KeyError(attempt)
        component = self.operations[row["component"]]
        resolver = getattr(component, "reconcile", None)
        if row["status"] not in ("issued", "unknown") or resolver is None:
            evidence = self.data.put({"attempt": attempt, "status": row["status"]}, self._json)
            status = row["status"] if row["status"] in ("completed", "failed") else "unknown"
            return Reconciliation(attempt, status, evidence)
        # Only the owning adapter can verify a native terminal receipt. Arbitrary
        # caller files or a transport timeout are not terminal evidence.
        terminal = resolver(attempt, dict(row), self.data)
        if terminal.attempt != attempt or terminal.status not in ("completed", "failed", "unknown"):
            raise ValueError("Invalid reconciliation")
        proof = self.data.get(terminal.evidence, self._json)
        if proof.get("binding") != row["binding"]:
            raise ContractConflict("Terminal evidence does not match the issued request")
        if terminal.status != "unknown":
            rng = _blob(proof["rng"])
            receipt = self.data.get(rng, design_codec)
            if (not isinstance(receipt, RNGReceipt) or receipt.before != _blob(row["before"])
                    or receipt.protocol != self.scientific_contract):
                raise ContractConflict("Terminal RNG receipt does not match the issued attempt")
            self.data.get(receipt.after, design_codec)
            result = _blob(proof.get("result"))
            if terminal.status == "completed":
                self.data.get(result, component.descriptor.result_codec)
            self._durable(terminal.evidence, rng, result)
            row.update(status=terminal.status, result=_blob_record(result), rng=_blob_record(rng),
                       error=proof.get("error"))
            self._write("attempt", attempt, row)
        return terminal

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
