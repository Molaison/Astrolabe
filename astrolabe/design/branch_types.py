"""Typed control records for reusable isolated branching compositions.

Members refer to owner-coded scientific records. A process/job observation is
not relabelled as a molecular Candidate. Molecular owners can retain complete
Candidates, Evidence and branch context inside their registered value codec.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .types import Blob, Key


@dataclass(frozen=True)
class BranchMember:
    id: int
    key: str
    value: Blob
    score: float
    eligible: bool = True
    comparison_context: Key | None = None


@dataclass(frozen=True)
class BranchProposal:
    id: int
    parent_id: int
    key: str
    value: Blob
    valid: bool = True
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class BranchJob:
    phase: Literal["propose", "evaluate"]
    id: int
    generation: int
    seed: int
    parent: BranchMember | None = None
    proposal: BranchProposal | None = None
    slot: int = 0
    source: Blob | None = None

    @property
    def identity(self) -> str:
        return f"{self.phase}/{self.id}"


@dataclass(frozen=True)
class BranchTerminal:
    job: BranchJob
    status: Literal["completed", "refused"]
    value: BranchMember | BranchProposal | None = None


@dataclass(frozen=True)
class BranchingState:
    generation: int = 0
    proposals: int = 0
    next_evaluation: int = 0
    records: tuple[BranchMember, ...] = ()
    frontier: tuple[BranchMember, ...] = ()
    parents: tuple[BranchMember, ...] = ()
    counts: tuple[tuple[int, int], ...] = ()
    batch: tuple[BranchJob, ...] = ()
    batch_results: tuple[BranchTerminal, ...] = ()
    queued: tuple[BranchJob, ...] = ()
    pending: tuple[BranchJob, ...] = ()
    seen: tuple[str, ...] = ()
    duplicates: int = 0
    rejected: int = 0
    refused: tuple[BranchJob, ...] = ()
    stop_reason: str | None = None
    finished: bool = False


@dataclass(frozen=True)
class BranchAction:
    state: BranchingState
    jobs: tuple[BranchJob, ...] = ()
    wait: Literal["any", "all", "none"] = "any"


@dataclass(frozen=True)
class BranchEvent:
    terminals: tuple[BranchTerminal, ...]


def branch_codec(value_type):
    from .store import DataclassCodec
    return DataclassCodec(value_type,
        f"astrolabe.design.branching.{value_type.__name__}/v1",
        nested_types=(BranchMember, BranchProposal, BranchJob, BranchTerminal,
                      BranchingState, BranchAction, BranchEvent))
