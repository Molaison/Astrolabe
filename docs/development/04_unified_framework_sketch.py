"""Frozen target interface sketch, not a runtime module.

这是目标接口草图；当前运行实现与迁移边界见 06_unified_framework_implementation.md。

Frozen signature contract; proposed API, no runtime implementation.

All referenced blobs/codecs are content-immutable. Keys are canonical digests,
not Python identities. Concrete requests/results belong to their family owner.
Operation is a durable invocation boundary, never a universal payload protocol.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, Protocol, TypeVar

I = TypeVar('I')
O = TypeVar('O')
T = TypeVar('T')
S = TypeVar('S')
@dataclass(frozen=True)
class Key:
    digest: str
@dataclass(frozen=True)
class Blob:
    identity: Key
    codec: str
@dataclass(frozen=True)
class Polymer:
    kind: Literal['protein', 'rna', 'dna']
    tokens: tuple[str, ...]
    modifications: tuple[tuple[int, str], ...]
@dataclass(frozen=True)
class Chemical:
    graph: Blob  # canonical atom IDs, elements, bonds, stereo, charge, attachments
@dataclass(frozen=True)
class Assignment:
    variable: str
    value: Polymer | Chemical
@dataclass(frozen=True)
class Occurrence:
    identity: Key
    state: str
    assembly: str
    chain: str
    component: str
    variable: str
    index_map: Blob  # ordered occurrence residue/atom -> variable token/atom IDs
@dataclass(frozen=True)
class Domain:
    variable: str
    fixed_tokens: tuple[int, ...]
    fixed_atoms: tuple[str, ...]
    allowed_values: Blob | None  # typed family-owned restrictions codec
@dataclass(frozen=True)
class System:
    identity: Key
    variables: tuple[str, ...]
    occurrences: tuple[Occurrence, ...]
    topology: Blob  # cross-variable/copy bonds and molecular connectivity
    domains: tuple[Domain, ...]  # explicit domains; referenced restrictions use DataStore
    def edit(self, base: Candidate, replacements: tuple[Assignment, ...], *, data: DataStore | None = None) -> Candidate:
        raise NotImplementedError
@dataclass(frozen=True)
class Candidate:
    identity: Key  # complete assignments + System identity, excludes lineage
    system: Key
    assignments: tuple[Assignment, ...]
@dataclass(frozen=True)
class Dependency:
    role: str
    identity: Key
@dataclass(frozen=True)
class InputManifest:
    identity: Key
    dependencies: tuple[Dependency, ...]  # actual consumed inputs, not paths alone
@dataclass(frozen=True)
class Sample:
    protocol: Key
    replicate: str  # independent requested attempts never collapse on seed equality
    seed: int
    index: str
@dataclass(frozen=True)
class Artifact:
    identity: Key
    candidate: Key
    occurrences: tuple[Occurrence, ...]
    sample: Sample
    content: Blob
    derived_from: tuple[Key, ...]
    inputs: InputManifest
@dataclass(frozen=True)
class Axis:
    name: str  # unique within a measurement, e.g. left_site/right_site
    kind: str  # extensible semantic kind; ordered pairs use two separate axes
    index_map: Blob  # ordered labels + explicit residue/atom/occurrence mapping
@dataclass(frozen=True)
class Subject:
    candidate: Key
    state: str | None
    assembly: str | None
    occurrences: tuple[Key, ...]
    artifact: Key | None
    sample: Sample | None
@dataclass(frozen=True)
class Measurement:
    identity: Key
    quantity: str  # producer-qualified name
    definition: Key
    units: str
    producer: Key
    subjects: tuple[Subject, ...]  # measured entity/relation, not an implicit array axis
    axes: tuple[Axis, ...]
    values: Blob | None  # natural shape: scalar (), site (L,), pair (L,R)
    missing: Literal['present', 'partial', 'unavailable', 'failed']
    valid_mask: Blob | None  # mandatory for partial values; same mapped shape
    inputs: InputManifest
@dataclass(frozen=True)
class Selection:
    subject: Subject
    axis_name: str
    index_map: Blob
    mask: Blob
@dataclass(frozen=True)
class Reduction:
    axes: tuple[str, ...]  # unique axis names, not ambiguous kinds
    method: str  # owner-registered reducer, e.g. weighted_mean
    parameters: Blob
    missing: Literal['fail', 'omit', 'propagate']

class MeasurementView(Protocol):
    """Immutable view validates subjects, axis maps, missingness and units.

    Reductions return derived measurements with lineage; raw data is retained.
    between selects ordered axes; it does not recompute isolated-region physics.
    """
    def select(self, selection: Selection) -> MeasurementView: raise NotImplementedError
    def between(self, left: Selection, right: Selection, *,
                left_axis: str, right_axis: str) -> MeasurementView: raise NotImplementedError
    def reduce(self, reduction: Reduction) -> Measurement: raise NotImplementedError
    def materialize(self) -> Measurement: raise NotImplementedError
@dataclass(frozen=True)
class Evidence:
    identity: Key
    evaluated_candidate: Key
    artifacts: tuple[Artifact, ...]
    measurements: tuple[Measurement, ...]
    inputs: InputManifest
    def with_measurements(self, measurements: tuple[Measurement, ...]) -> Evidence:
        raise NotImplementedError
@dataclass(frozen=True)
class EvidenceUse:
    evidence: Evidence
    relation: Literal['current', 'base_guidance', 'reference']
    working_candidate: Key  # does not rewrite evaluated_candidate or subjects
@dataclass(frozen=True)
class Proposal:
    identity: Key
    parents: tuple[Key, ...]
    base: Key
    candidate: Candidate
    used_evidence: tuple[Key, ...]
    protocol: Key
@dataclass(frozen=True)
class Capability:
    namespace: str
    name: str
    version: str
@dataclass(frozen=True)
class Requirement:
    component: str
    capability: Capability
    enforcement: Literal['required', 'preferred']
@dataclass(frozen=True)
class SelectedOptionsSchema:
    owner: str
    version: str
    json_schema: str  # readable canonical JSON Schema, loaded with registration
    effective_options_json: str  # validated canonical selected options, not digest
@dataclass(frozen=True)
class ComponentDescriptor(Generic[I, O]):
    identity: Key  # implementation/version, effective options and model identity
    alias: str
    request_type: type[I]
    result_type: type[O]
    options_schema: SelectedOptionsSchema
    request_codec: Codec[I]
    result_codec: Codec[O]
    capabilities: tuple[Capability, ...]

class Operation(Protocol[I, O]):
    descriptor: ComponentDescriptor[I, O]
    # Private adapter preparation/execution is NOT exposed in the public API.
@dataclass(frozen=True)
class Conditioning:
    identity: Key
    owner: Key
    intent: Blob  # owner's typed intent codec, not a global Intent.kind enum
    mechanism: str
    effective_parameters: Blob
    approximations: tuple[str, ...]
    inputs: InputManifest
@dataclass(frozen=True)
class InitializeRequest:
    system: System
    sample: Sample
@dataclass(frozen=True)
class Initialized:
    candidate: Candidate
    evidence: Evidence | None
@dataclass(frozen=True)
class PredictRequest:
    system: System
    candidate: Candidate
    assemblies: tuple[str, ...]
    conditioning: tuple[Conditioning, ...]
    sources: tuple[Dependency, ...]
    samples: tuple[Sample, ...]
@dataclass(frozen=True)
class TransformRequest:
    artifact: Artifact
@dataclass(frozen=True)
class ArtifactBank:
    artifacts: tuple[Artifact, ...]
@dataclass(frozen=True)
class ChooseArtifactRequest:
    bank: ArtifactBank
    index: int  # explicit policy; no silent first-result discard
@dataclass(frozen=True)
class SelectRequest:
    system: System
    candidate: Candidate
    evidence: tuple[EvidenceUse, ...]
@dataclass(frozen=True)
class PackRequest:
    artifacts: tuple[Artifact, ...]
    selections: tuple[Selection, ...]
    ties: Blob
    state_weights: tuple[float, ...]
@dataclass(frozen=True)
class PackedStates:
    artifact: Artifact
    mapping: Blob
    ties: Blob
    state_weights: tuple[float, ...]
@dataclass(frozen=True)
class SamplingPolicy:
    identity: Key
    options: Blob  # typed sampling owner's bias/mask/temperature/count schema
@dataclass(frozen=True)
class Applied:
    request: Key
    operation: str
    parameters: Key
    decoding_boundary: str
@dataclass(frozen=True)
class IndependentDistribution:
    identity: Key
    alphabet: tuple[str, ...]
    sites: Blob
    samples: tuple[Sample, ...]
    values: Blob
    semantics: Literal['logits', 'log_probs', 'sampling_probs', 'empirical']
    conditioning: InputManifest
    applied: tuple[Applied, ...]
    ancillary_evidence: tuple[Key, ...]  # primary-only logs stay primary-only
@dataclass(frozen=True)
class ProposeRequest:
    system: System
    parents: tuple[Candidate, ...]
    working: Candidate
    evidence: tuple[EvidenceUse, ...]
    selection: tuple[Selection, ...]
    packed: PackedStates | None
    sampling: SamplingPolicy
    sample: Sample
@dataclass(frozen=True)
class Proposals:
    items: tuple[Proposal, ...]  # direct/joint sampler returns complete proposals
@dataclass(frozen=True)
class SampleIndependentRequest:
    distribution: IndependentDistribution
    context: ProposeRequest
    count: int
@dataclass(frozen=True)
class FuseRequest:
    distributions: tuple[IndependentDistribution, ...]
    weights: tuple[float, ...]
@dataclass(frozen=True)
class MeasureRequest:
    system: System
    candidate: Candidate
    evidence: tuple[EvidenceUse, ...]
    selections: tuple[Selection, ...]
@dataclass(frozen=True)
class AssessRequest:
    candidate: Candidate
    evidence: Evidence
    history: History
    objective_revision: Key
@dataclass(frozen=True)
class Assessment:
    evidence: Key
    objective_revision: Key
    history_revision: Key
    values: tuple[float, ...]
    gates: tuple[tuple[str, bool], ...]
    inputs: InputManifest
@dataclass(frozen=True)
class Scored:
    candidate: Candidate
    evidence: Evidence
    assessment: Assessment
@dataclass(frozen=True)
class Work:
    identity: str
    parents: tuple[Scored, ...]
    stage: str
    sample: Sample
@dataclass(frozen=True)
class TrialOutcome:
    work: str
    status: Literal['completed', 'failed', 'unknown', 'refused']
    proposals: tuple[Proposal, ...]
    children: tuple[Scored, ...]
    attempts: tuple[str, ...]
@dataclass(frozen=True)
class History:
    revision: Key
    records: tuple[TrialOutcome, ...]
@dataclass(frozen=True)
class CallKey:
    work_id: str
    named_round: str  # hierarchical stable path, not completion-order counter
    component: Key
    input_manifest: Key  # private journal/recovery record; runtime computes
@dataclass(frozen=True)
class RNGReceipt:
    before: Blob
    after: Blob
    protocol: Key
@dataclass(frozen=True)
class Attempt:
    identity: str
    key: CallKey
    status: Literal['reserved', 'issued', 'completed', 'failed', 'unknown']
    result: Blob | None
    rng: RNGReceipt | None
@dataclass(frozen=True)
class Reconciliation:
    attempt: str
    status: Literal['completed', 'failed', 'unknown']
    evidence: Blob  # validated owner/terminal-record evidence

class Codec(Protocol[T]):
    identity: str
    def encode(self, value: T) -> bytes: raise NotImplementedError
    def decode(self, data: bytes) -> T: raise NotImplementedError
    def references(self, value: T) -> tuple[Blob, ...]: raise NotImplementedError

class DataStore(Protocol):
    """One content-addressed owner; codec implementations belong to families.

    put canonically encodes/hashes into memory, with no filesystem writes.
    get checks codec identity/content digest. persist durably commits the full
    referenced closure using registered versioned codecs before journal commit.
    Dry-run uses memory-only put/get and never persist. Unknown codecs fail.
    """
    def put(self, value: T, codec: Codec[T]) -> Blob: raise NotImplementedError
    def get(self, reference: Blob, codec: Codec[T]) -> T: raise NotImplementedError
    def persist(self, references: tuple[Blob, ...]) -> None: raise NotImplementedError
@dataclass(frozen=True)
class Preflight:
    identity: Key
    requirements: tuple[Requirement, ...]
    scientific_contract: Key

class Runtime(Protocol):
    """One owner for binding, deps, admission, cache, attempts, RNG and outputs.

    Work-entry binds work_id/history. call privately binds/prepares/validates
    its CallKey; callers supply only a stable named_round.
    Complete results+RNG receipts commit before return. Resume repeats control
    flow and restores completed calls' post-RNG state without model execution.
    Unknown blocks replay; explicit retry is a NEW attempt with prior lineage.
    """
    history: History
    data: DataStore
    def call(self, named_round: str, operation: Operation[I, O], request: I) -> O: raise NotImplementedError
    def preflight(self, requirements: tuple[Requirement, ...]) -> Preflight: raise NotImplementedError
    def reconcile(self, attempt: str) -> Reconciliation: raise NotImplementedError

class Trial(Protocol):
    requirements: tuple[Requirement, ...]  # ALL possible operations, before loading
    def root(self, candidate: Candidate, runtime: Runtime) -> Scored: raise NotImplementedError
    def perform(self, work: Work, runtime: Runtime) -> TrialOutcome: raise NotImplementedError

class SearchPolicy(Protocol[S]):
    codec: Codec[S]
    def start(self, initial: tuple[Scored, ...]) -> S: raise NotImplementedError
    def ask(self, state: S, history: History, capacity: int) -> tuple[Work, ...]: raise NotImplementedError
    def tell(self, state: S, outcomes: tuple[TrialOutcome, ...]) -> S: raise NotImplementedError
    def finished(self, state: S) -> bool: raise NotImplementedError
    def selected(self, state: S) -> tuple[Scored, ...]: raise NotImplementedError
@dataclass(frozen=True)
class RunManifest:
    scientific_contract: Key
    preflight: Key
    policy_codec: str
    result_codec: str
    journal: Path
    checkpoint: Blob
    selected_summary: Path
@dataclass(frozen=True)
class RunResult:
    selected: tuple[Scored, ...]
    manifest: RunManifest

class Components(Protocol):
    @classmethod
    def from_config(cls, config: Path) -> Components: raise NotImplementedError
    def operation(self, alias: str, request: type[I], result: type[O]) -> Operation[I, O]: raise NotImplementedError
    def sampling(self, alias: str) -> SamplingPolicy: raise NotImplementedError
    def requirements(self, aliases: tuple[str, ...]) -> tuple[Requirement, ...]: raise NotImplementedError

def run(system: System, initial: Candidate, trial: Trial,
        policy: SearchPolicy[S], runtime: Runtime) -> RunResult:
    raise NotImplementedError

# run calls runtime.preflight(trial.requirements) once before any model loading.
# Owner contracts (mandatory implementation invariants):
# Registration attaches descriptors/codecs to actual family registries. New
# families add typed requests/results+an owning registry, not central branches.
# Runtime-private adapters bind/prepare and expose actual input manifests,
# paid-operation accounting boundaries and resource requirements to Runtime.
# Runtime rejects undeclared components/capabilities before invoking an adapter.
# Dynamic request validation remains mandatory even after Trial preflight.
# No cached replicate substitution: call replay restores the SAME attempt;
# reuse across candidates needs explicit projected-input equality + new binding.
# Confidence subjects remain original predictions after coordinate transforms.
# Every repeated call key verifies request/preparation identity; changed input
# at an existing work/round slot is a contract conflict, not fresh paid work.
# reserved -> issued -> completed|failed|unknown. Terminal results and receipts
# are durable before return. Crash between issue and terminal commit is unknown.
# Failures/unknown retain budget per protocol; unknown never auto-refunds/replays.
# Retry requires explicit authorization/new attempt lineage; not another round.
# Trial rerun begins from its recorded entry candidate/history/RNG checkpoint.
# Calls restore recorded RNG receipts; non-operation randomness/side effects and
# mutable external reads inside Trial are forbidden. History is an immutable
# work-entry snapshot, not live wall-clock-dependent state.
# Result codecs and policy checkpoint codecs are versioned in the run manifest.
# Live gradient tensors never pass through Blob as a fake persistent session.
# An optional real adapter owns typed gradient requests/results and tensor
# lifetimes internally; no capability means preflight failure, not fallback.

# Private registry adapter seam; not re-exported to scientific callers.
@dataclass(frozen=True)
class _Prepared(Generic[I]):
    request: I
    inputs: InputManifest
    requirements: tuple[Requirement, ...]
    paid_boundary: str

class _Adapter(Protocol[I, O]):
    def prepare(self, request: I) -> _Prepared[I]: raise NotImplementedError
    def invoke(self, prepared: _Prepared[I]) -> O: raise NotImplementedError

def measurement_view(value: Measurement, data: DataStore) -> MeasurementView:
    raise NotImplementedError
