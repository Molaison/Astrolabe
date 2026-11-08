"""Scientific mutator projections committed by the existing Runtime owner.

These records do not authorize replay. An issued proposal without a terminal
receipt remains unknown even when its post-call state was saved successfully.
"""
from __future__ import annotations

from copy import copy

from .runtime import ContractConflict, OutcomeUnknown
from .store import JSONCodec

STATEFUL_MUTATORS = frozenset({"dplm", "pore_blocker_mcmc", "rfd4"})
_CODEC = JSONCodec("astrolabe.configured-refinement.mutator-state/v1")
_NAME = "configured-refinement/mutator-state"


class MutatorStateJournal:
    def __init__(self, runtime, protocol, aliases):
        self.runtime = runtime
        self.protocol = protocol.digest
        self.aliases = tuple(aliases)
        self.runtime.data.register(_CODEC)
        self._record = self.runtime.load(_NAME, _CODEC) if self.aliases else None
        if self.aliases:
            prior = tuple(
                attempt for attempt in self.runtime.attempts()
                if attempt.key.named_round.startswith("native-refinement/propose/")
                and attempt.key.named_round.endswith("/proposal")
                and attempt.status != "reserved"
            )
            if any(attempt.status in {"issued", "unknown"} for attempt in prior):
                raise OutcomeUnknown("Mutator proposal is unresolved; state cannot authorize replay")
            if prior and self._record is None:
                raise ContractConflict("Completed proposal has no mutator scientific state")
            self._validate(self._record)
            if prior and self._record["step"] < max(
                int(attempt.key.named_round.split("/")[-2]) for attempt in prior
            ):
                raise ContractConflict("Mutator scientific state is older than its proposal receipts")

    def _validate(self, record):
        if record is None:
            return
        if (not isinstance(record, dict) or type(record.get("version")) is not int
                or record.get("version") != 1 or record.get("protocol") != self.protocol
                or tuple(record.get("aliases", ())) != self.aliases
                or set(record.get("states", {})) != set(self.aliases)
                or type(record.get("step")) is not int or record["step"] < 0):
            raise ContractConflict("Mutator scientific state belongs to another contract")

    def restore(self, mutators):
        if not self.aliases:
            return
        record = self._record
        if record is None:
            self.commit(mutators, step=0)
            return
        # Supported adapters replace their scientific attributes only; they
        # neither load models nor mutate shared native handles. Validate all
        # projections on shallow owner views before publishing any of them.
        staged = []
        for alias in self.aliases:
            owner = mutators[alias]
            prepared = copy(owner)
            prepared.restore_state(record["states"][alias])
            staged.append((owner, prepared))
        for owner, prepared in staged:
            owner.__dict__.update(prepared.__dict__)

    def commit(self, mutators, *, step):
        if not self.aliases:
            return
        try:
            states = {alias: mutators[alias].export_state() for alias in self.aliases}
            record = {
                "version": 1, "protocol": self.protocol,
                "aliases": list(self.aliases), "step": step, "states": states,
            }
            self.runtime.save(_NAME, record, _CODEC)
            self._record = record
        except BaseException as error:
            # An effect may have happened even if the state publication failed.
            raise OutcomeUnknown("Mutator scientific state was not committed; do not replay proposal") from error
