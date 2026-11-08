"""Composable design: immutable scientific data, typed calls and search.

Importing this module never constructs a model. Scientific families register
their own adapters; the coordinator has no model or molecular task branches.
"""
from .types import (
    Artifact, Assignment, Axis, Blob, Candidate, Capability, Chemical,
    Conditioning, Continuation, Dependency, Domain, Evidence, EvidenceUse, History, Key,
    Measurement, Occurrence, Polymer, Proposal, Reduction, Requirement,
    Sample, Scored, Selection, Subject, System, TrialOutcome, Work,
    key_for, make_candidate, make_system, manifest,
)
from .store import ArrayCodec, DataStore, DataclassCodec, JSONCodec
from .operations import Components, FunctionOperation, operation
from .runtime import Runtime
from .coordinator import run
from .composition import ControlRun, DesignRun

__all__ = [
    "Artifact", "Assignment", "Axis", "Blob", "Candidate", "Capability",
    "Chemical", "Components", "Conditioning", "Continuation", "ControlRun", "DataStore", "DataclassCodec", "DesignRun",
    "Dependency", "Domain", "Evidence", "EvidenceUse", "FunctionOperation",
    "History", "JSONCodec", "ArrayCodec", "Key", "Measurement", "Occurrence",
    "Polymer", "Proposal", "Reduction", "Requirement", "Runtime", "Sample",
    "Scored", "Selection", "Subject", "System", "TrialOutcome", "Work",
    "key_for", "make_candidate", "make_system", "manifest", "operation", "run",
]
