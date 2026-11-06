"""Typed, model-free support for protein branching tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

import astrolabe.protein_beam as beam
from astrolabe.design.branch_execution import BranchExecutor
from astrolabe.design.branch_types import (
    BranchAction,
    BranchEvent,
    BranchJob,
    BranchMember,
    BranchProposal,
    BranchTerminal,
    BranchingState,
)
from astrolabe.design.branching import BranchingPolicy
from astrolabe.design.composition import ControlRun
from astrolabe.design.store import DataStore, JSONCodec
from astrolabe.design.types import Blob, Key, key_for


FIXTURE_PATH = Path(__file__).parent / "data/protein_beam/02_fixture.yaml"
CONTEXT = key_for(("branching-test-support/comparison-context/v1",))


@dataclass(frozen=True)
class NativeControlAssets:
    af3_dir: Path
    ligandmpnn_dir: Path
    af3_asset: Path
    ligandmpnn_asset: Path


@pytest.fixture
def native_control_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> NativeControlAssets:
    """Create path-only assets for native configuration and contract tests."""

    af3_dir = tmp_path / "assets/af3"
    ligand_dir = tmp_path / "assets/ligandmpnn"
    af3_dir.mkdir(parents=True)
    ligand_dir.mkdir(parents=True)
    af3 = af3_dir / "af3.bin"
    af3.write_bytes(b"fake CPU test AF3 asset; never loaded")
    names = (
        "ligandmpnn_v_32_020_25.pt",
        "proteinmpnn_v_48_020.pt",
        "solublempnn_v_48_020.pt",
    )
    for name in names:
        (ligand_dir / name).write_bytes(
            f"fake CPU test MPNN asset {name}; never loaded".encode("ascii")
        )
    monkeypatch.setenv("GH_WEIGHT_AF3", str(af3_dir))
    monkeypatch.setenv("GH_WEIGHT_LIGANDMPNN", str(ligand_dir))
    return NativeControlAssets(
        af3_dir=af3_dir,
        ligandmpnn_dir=ligand_dir,
        af3_asset=af3,
        ligandmpnn_asset=ligand_dir / "ligandmpnn_v_32_020_25.pt",
    )


def native_branch_config():
    """Load the maintained fixture without altering its scientific values."""

    return beam.load_config(FIXTURE_PATH)


def _raw_blob(data: DataStore, codec: JSONCodec, raw: Mapping) -> Blob:
    return data.put(dict(raw), codec)


def _member(
    data: DataStore,
    codec: JSONCodec,
    raw: Mapping,
) -> BranchMember:
    return BranchMember(
        id=int(raw["id"]),
        key=str(raw.get("sequence", raw.get("key", f"parent-{raw['id']}"))),
        value=_raw_blob(data, codec, raw),
        score=float(raw.get("score", 0.0)),
        eligible=bool(raw.get("eligible", True)),
        comparison_context=CONTEXT,
    )


def _proposal(
    data: DataStore,
    codec: JSONCodec,
    job: BranchJob,
    raw: Mapping,
) -> BranchProposal:
    return BranchProposal(
        id=job.id,
        parent_id=job.parent.id,
        key=str(raw["sequence"]),
        value=_raw_blob(data, codec, raw),
        valid=bool(raw.get("valid", True)),
        reasons=tuple(raw.get("proposal_rejections", ())),
    )


@dataclass(frozen=True)
class PreparedGeneration:
    config: object
    policy: BranchingPolicy
    state: BranchingState
    parents: tuple[Mapping, ...]
    data: DataStore
    codec: JSONCodec
    context: Key

    def proposal_terminal(
        self,
        job: BranchJob,
        sequence: str,
        *,
        valid: bool = True,
        reasons: tuple[str, ...] = (),
    ) -> BranchTerminal:
        raw = {
            "id": job.id,
            "kind": "proposal",
            "parent_id": job.parent.id,
            "generation": job.generation,
            "seed": job.seed,
            "sequence": sequence,
            "valid": valid,
            "proposal_rejections": list(reasons),
        }
        return BranchTerminal(
            job=job,
            status="completed",
            value=_proposal(self.data, self.codec, job, raw),
        )

    def evaluation_terminal(
        self,
        job: BranchJob,
        *,
        score: float | None = None,
    ) -> BranchTerminal:
        raw = {
            "id": job.id,
            "kind": "evaluate",
            "generation": job.generation,
            "parent_id": job.parent.id,
            "proposal_id": job.proposal.id,
            "sequence": job.proposal.key,
            "seed": job.seed,
            "status": "done",
            "score": float(job.id if score is None else score),
            "metrics": {metric: 0.0 for metric in beam.METRICS},
        }
        return BranchTerminal(
            job=job,
            status="completed",
            value=_member(self.data, self.codec, raw),
        )

    def project_records(self) -> tuple[dict, ...]:
        return tuple(
            self.data.get(member.value, self.codec)
            for member in self.state.records
        )


def _root_record(
    config,
    root_defaults: Mapping | None,
) -> dict:
    raw = {
        "id": 0,
        "parent_id": None,
        "generation": 0,
        "sequence": "".join(str(config.seqs.A).replace("+", "").split()),
        "snapshot": "test-only-root-snapshot",
        "status": "done",
        "score": 0.0,
        "metrics": {metric: 0.0 for metric in beam.METRICS},
        "eligible": True,
    }
    raw.update(dict(root_defaults or {}))
    return raw


def prepare_one_generation(
    config,
    *,
    parents: Sequence[Mapping] | None = None,
    root_defaults: Mapping | None = None,
    width: int | None = None,
    proposals_per_parent: int | None = None,
    max_evaluations: int | None = None,
    proposal_cap: int | None = None,
    proposal_batch_size: int | None = None,
    evaluation_width: int | None = None,
    pipeline: bool | None = None,
    quota_mode: str = "admitted",
) -> PreparedGeneration:
    """Create generation 1 from already evaluated root/parent records."""

    parent_records = tuple(
        dict(value)
        for value in (
            parents if parents is not None else (_root_record(config, root_defaults),)
        )
    )
    if not parent_records:
        raise ValueError("at least one parent record is required")
    data = DataStore()
    codec = JSONCodec("branching-test-support/native-record/v1")
    data.register(codec)
    members = tuple(
        _member(data, codec, record) for record in parent_records
    )
    settings = config.protein_beam
    budget = int(
        max_evaluations
        if max_evaluations is not None
        else settings.max_evaluations
    )
    if budget <= len(members):
        raise ValueError("test generation needs evaluation room after its roots")
    policy = BranchingPolicy(
        width=width if width is not None else len(members),
        proposals_per_parent=(
            proposals_per_parent
            if proposals_per_parent is not None
            else int(settings.proposals_per_parent)
        ),
        max_evaluations=budget,
        proposal_cap=(
            proposal_cap if proposal_cap is not None else budget * 10
        ),
        seed=int(config.seed),
        evaluation_seed=int(config.seed),
        quota_mode=quota_mode,
        expansion="balanced",
        proposal_batch_size=(
            proposal_batch_size
            if proposal_batch_size is not None
            else int(settings.get("proposal_batch_size", 3))
        ),
        evaluation_width=(
            evaluation_width
            if evaluation_width is not None
            else int(settings.get("in_flight", 2))
        ),
        pipeline=(
            bool(settings.get("pipeline_proposals", False))
            if pipeline is None
            else pipeline
        ),
        max_generations=1,
    )
    state = BranchingState(
        generation=1,
        next_evaluation=max(member.id for member in members) + 1,
        records=members,
        frontier=members,
        parents=members,
        seen=tuple(member.key for member in members),
    )
    return PreparedGeneration(
        config=config,
        policy=policy,
        state=state,
        parents=parent_records,
        data=data,
        codec=codec,
        context=CONTEXT,
    )


@dataclass(frozen=True)
class GenerationResult:
    prepared: PreparedGeneration
    state: BranchingState
    proposals: tuple[Mapping, ...]
    evaluations: tuple[Mapping, ...]
    submitted: tuple[str, ...]
    errors: tuple[tuple[str, str], ...]

    def project_records(self) -> tuple[dict, ...]:
        return tuple(
            self.prepared.data.get(member.value, self.prepared.codec)
            for member in self.state.records
        )


def run_one_generation(
    prepared: PreparedGeneration,
    *,
    proposal: Callable[[BranchJob, Mapping], Mapping] | None = None,
    evaluation: Callable[[BranchJob, Mapping], Mapping] | None = None,
    controlled: Callable[[BranchJob], None] | None = None,
    workers: int = 4,
) -> GenerationResult:
    """Run one bounded generation through the common policy/executor kernel."""

    parent_records = {
        int(record["id"]): record for record in prepared.parents
    }
    proposal_records: dict[int, Mapping] = {}
    evaluation_records: dict[int, Mapping] = {}
    submitted: list[str] = []
    errors: list[tuple[str, str]] = []

    def worker(job: BranchJob):
        if controlled is not None:
            controlled(job)
        if job.phase == "propose":
            parent = parent_records[job.parent.id]
            sequence = list(str(parent["sequence"]))
            sequence[job.id % len(sequence)] = "ACDEFGHIKLMNPQRSTVWY"[
                job.id % 20
            ]
            raw = {
                "id": job.id,
                "kind": "proposal",
                "parent_id": job.parent.id,
                "generation": job.generation,
                "seed": job.seed,
                "sequence": "".join(sequence),
                "valid": True,
                "proposal_rejections": [],
            }
            if proposal is not None:
                raw.update(dict(proposal(job, parent)))
            return raw

        proposal_raw = prepared.data.get(job.proposal.value, prepared.codec)
        raw = {
            "id": job.id,
            "kind": "evaluate",
            "generation": job.generation,
            "parent_id": job.parent.id,
            "proposal_id": job.proposal.id,
            "sequence": job.proposal.key,
            "seed": job.seed,
            "status": "done",
            "score": float(job.id),
            "metrics": {metric: 0.0 for metric in beam.METRICS},
        }
        if evaluation is not None:
            raw.update(dict(evaluation(job, proposal_raw)))
        return raw

    def resolve(job: BranchJob, raw) -> BranchTerminal:
        if not isinstance(raw, Mapping):
            raise TypeError("test native worker must return a mapping")
        if raw.get("refused"):
            return BranchTerminal(job, "refused")
        if job.phase == "propose":
            proposal_records[job.id] = dict(raw)
            return BranchTerminal(
                job,
                "completed",
                _proposal(prepared.data, prepared.codec, job, raw),
            )
        if raw.get("status") != "done":
            raise ValueError("test evaluation must return a terminal done record")
        evaluation_records[job.id] = dict(raw)
        return BranchTerminal(
            job,
            "completed",
            _member(prepared.data, prepared.codec, raw),
        )

    def submit(pool, job):
        submitted.append(job.identity)
        return pool.submit(worker, job)

    def on_error(job, error):
        errors.append((job.identity, type(error).__name__))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        executor = BranchExecutor(
            pool,
            submit=submit,
            resolve=resolve,
            on_error=on_error,
        )
        state = ControlRun(
            prepared.policy,
            executor,
            prepared.state,
        ).run()
    return GenerationResult(
        prepared=prepared,
        state=state,
        proposals=tuple(
            proposal_records[key] for key in sorted(proposal_records)
        ),
        evaluations=tuple(
            evaluation_records[key] for key in sorted(evaluation_records)
        ),
        submitted=tuple(submitted),
        errors=tuple(errors),
    )


__all__ = [
    "CONTEXT",
    "FIXTURE_PATH",
    "GenerationResult",
    "NativeControlAssets",
    "PreparedGeneration",
    "native_branch_config",
    "native_control_assets",
    "prepare_one_generation",
    "run_one_generation",
]
