from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class ResidueRef:
    """A physical chain and its one-based residue ordinal, not author numbering."""

    chain: str
    position: int


@dataclass(frozen=True, order=True)
class AtomRef:
    chain: str
    position: int
    atom: str


@dataclass(frozen=True)
class ChainSpec:
    chain: str
    protomer: str
    sequence: str
    kind: str = "protein"

    @property
    def residue_count(self):
        return len(self.sequence) if self.kind == "protein" else 1


@dataclass(frozen=True)
class HBondCondition:
    atom: AtomRef
    role: str


@dataclass(frozen=True)
class ConditionPlan:
    released: tuple[ResidueRef, ...]
    fixed_coordinates: tuple[ResidueRef, ...]
    hbonds: tuple[HBondCondition, ...] = ()
    excluded_amino_acids: tuple[str, ...] = ()
    feedback_json: str = "{}"
    reconstruction: dict | None = None


@dataclass(frozen=True)
class GenerationRequest:
    structure_path: str
    assembly: str
    chains: tuple[ChainSpec, ...]
    plan: ConditionPlan
    seed: int


@dataclass(frozen=True)
class GeneratedStructure:
    structure_path: str
    annotated_input_path: str
    manifest_path: str
    sequences: tuple[tuple[str, str], ...]
    request_digest: str
    structure_digest: str
    cache_hit: bool


@dataclass(frozen=True)
class GenerationResult:
    assembly: str
    chains: tuple[ChainSpec, ...]
    generated: GeneratedStructure
