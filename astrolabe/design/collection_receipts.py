"""Durable receipt bridge for typed generation collection.

The collection policy and worker remain owned by ``collection_search``.  This
module only orders admission, terminal observation and collection commits
through the control thread, then maps owner-verified recovery into the shared
``ControlJournal`` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .collection import (
    CollectionCommit,
    CollectionConfig,
    Job,
    JobResult,
    ResultCollection,
    ResultSelector,
)
from .collection_search import (
    CollectionAction,
    CollectionEvent,
    CollectionTrial,
    collection_action_id,
)
from .control_journal import ControlRecovery
from .runtime import OutcomeUnknown
from .store import design_codec
from .types import Blob, Key, RNGReceipt, key_for


@dataclass(frozen=True)
class RehydratedCollection:
    ready_results: tuple[JobResult, ...] = ()
    issued_jobs: tuple[Job, ...] = ()
    uncommitted_generations: tuple[int, ...] = ()
    collection_index: int = 0


class CollectionReceiptBackend(Protocol):
    """Owner-specific receipts consumed by :class:`DurableCollectionTrial`."""

    def rehydrate(self) -> RehydratedCollection: ...

    def admission(self, job: Job) -> JobResult | None: ...

    def terminal_observed(self, job: Job, result: JobResult) -> None: ...

    def collection_commit(self, commit: CollectionCommit) -> None: ...

    def committed_collection(
        self,
        action_id: str,
    ) -> tuple[ResultCollection, Blob, Blob] | None: ...


class DurableCollectionTrial:
    """Recorded collection trial with explicit resume cuts."""

    def __init__(
        self,
        *,
        runtime,
        config: CollectionConfig,
        worker,
        selector: ResultSelector,
        receipts: CollectionReceiptBackend,
        action_codec,
        result_codec,
        logger=None,
        resumed: bool = False,
    ) -> None:
        self.runtime = runtime
        self.receipts = receipts
        self.action_codec = action_codec
        self.result_codec = result_codec
        self.resumed = bool(resumed)
        self._hydrated = False
        self._rehydrated_generations: frozenset[int] = frozenset()
        self._submitted_generations: set[int] = set()
        self.trial = CollectionTrial(
            config=config,
            worker=worker,
            selector=selector,
            logger=logger,
            admission=receipts.admission,
            terminal_observed=receipts.terminal_observed,
            collection_commit=receipts.collection_commit,
            rng_snapshot=runtime._capture_rng,
        )
        self.identity = key_for(
            (
                "astrolabe.durable-collection-trial/v1",
                self.trial.identity,
                action_codec.identity,
                result_codec.identity,
                getattr(receipts, "identity", None),
            )
        )

    def perform(self, action: CollectionAction) -> CollectionEvent:
        self._hydrate()
        action_id = collection_action_id(action)
        committed = self.receipts.committed_collection(action_id)
        if committed is not None:
            collection, _rng_before, rng_after = committed
            self.runtime._restore_rng(rng_after)
            return CollectionEvent(action=action, collections=(collection,))
        if (
            self.resumed
            and action.kind == "collect"
            and action.generation not in self._submitted_generations
            and self._has_collection_activity(action.generation)
        ):
            raise OutcomeUnknown(
                "Collection was interrupted before its selection boundary "
                "was durably committed"
            )
        result = self.trial.perform(action)
        if action.kind == "submit" and action.generation is not None:
            self._submitted_generations.add(int(action.generation))
        return result

    def recover(self, request: CollectionAction) -> ControlRecovery:
        """Reconcile only boundaries that have durable owner proof."""

        if not isinstance(request, CollectionAction):
            raise TypeError("Durable collection recovery requires CollectionAction")
        self._hydrate()
        action_id = collection_action_id(request)
        committed = self.receipts.committed_collection(action_id)
        if committed is not None:
            collection, rng_before, rng_after = committed
            self.runtime._restore_rng(rng_before)
            result = CollectionEvent(action=request, collections=(collection,))
            self.runtime._restore_rng(rng_after)
        elif request.kind == "submit":
            rng_before = self.runtime._capture_rng()
            result = self.trial.perform(request)
            rng_after = self.runtime._capture_rng()
            if rng_after != rng_before:
                raise OutcomeUnknown(
                    "Submit recovery consumed scientific RNG"
                )
            if request.generation is not None:
                self._submitted_generations.add(int(request.generation))
        else:
            raise OutcomeUnknown(
                "Collection request has no committed selection boundary"
            )

        request_ref = self.runtime.data.put(request, self.action_codec)
        contract = self.runtime.load("control/contract")
        if not isinstance(contract, Key):
            raise OutcomeUnknown("Control contract is unavailable for recovery")
        execution = key_for((contract, request_ref)).digest
        result_ref = self.runtime.data.put(result, self.result_codec)
        rng_ref = self.runtime.data.put(
            RNGReceipt(
                rng_before,
                rng_after,
                self.runtime.scientific_contract,
            ),
            design_codec,
        )
        return ControlRecovery(
            execution=Key(execution),
            request=request_ref,
            result=result_ref,
            rng_receipt=rng_ref,
        )

    def checkpoint(self, _state, *, request) -> None:
        # Per-job admission/terminal receipts and collection selection
        # receipts are the durable facts used by this owner.
        return None

    def close(self, *, wait_for_pending: bool = True) -> None:
        self.trial.close(wait_for_pending=wait_for_pending)

    def __enter__(self) -> DurableCollectionTrial:
        return self

    def __exit__(self, exc_type, _exc, _tb) -> bool:
        self.close(wait_for_pending=exc_type is None)
        return False

    def _hydrate(self) -> None:
        if self._hydrated:
            return
        snapshot = self.receipts.rehydrate()
        self._rehydrated_generations = frozenset(
            snapshot.uncommitted_generations
        )
        self.trial.rehydrate(
            ready_results=snapshot.ready_results,
            issued_jobs=snapshot.issued_jobs,
            collection_index=snapshot.collection_index,
        )
        self._hydrated = True

    def _has_collection_activity(self, generation: int | None) -> bool:
        if generation is None:
            return bool(self._rehydrated_generations)
        generation = int(generation)
        if generation in self._rehydrated_generations:
            return True
        pool = self.trial.pool
        return any(
            int(item.job.generation) == generation
            for item in pool._ready
        ) or any(
            int(item.generation) == generation for item in pool._pending.values()
        ) or any(
            int(item.generation) == generation for item in pool._issued.values()
        ) or any(
            int(item.generation) == generation
            for item in pool._known_jobs.values()
        )


__all__ = [
    "CollectionReceiptBackend",
    "DurableCollectionTrial",
    "RehydratedCollection",
]
