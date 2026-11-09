"""Read-only derivative export of a drained, proposal-cap-failed native run.

This is deliberately not resume, recovery, Score, or checkpoint migration. The
source control contract is retained as evidence, never rewritten or recomputed
from today's configured-refinement factory.
"""
from __future__ import annotations

from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from types import SimpleNamespace

from omegaconf import OmegaConf

from .branch_owner import _RECEIPT
from .branch_types import BranchAction, BranchEvent, BranchJob, BranchingState, branch_codec
from .branching import BranchingPolicy
from .configured_refinement import (
    _PREPARED_CHECKPOINT_CODEC, _asset_manifest_identity, _candidate_state,
    _generic_proposal_seed, _read_asset_manifest, _scientific_config_payload,
    _search_settings,
)
from .control_journal import _CHECKPOINT
from .io import atomic_json
from .native_refinement import NativeRefinementOwner, NativeRefinementResult
from .native_refinement_runtime import _REFERENCE_CODECS
from .native_support import ModelInitializationRequest, ModelReady
from .refinement_worker import _Worker, _blob as _worker_blob
from .runtime import ContractConflict, OutcomeUnknown, _blob
from .store import ArrayCodec, DataStore, DataclassCodec, JSONCodec, design_codec
from .types import Key, key_for


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data(root):
    data = DataStore(root)
    for codec in (
        *_REFERENCE_CODECS, design_codec, ArrayCodec(), JSONCodec(),
        _CHECKPOINT, _PREPARED_CHECKPOINT_CODEC, _RECEIPT,
        *(branch_codec(cls) for cls in (BranchAction, BranchEvent, BranchJob, BranchingState)),
        *(DataclassCodec(cls, f"{cls.__module__}.{cls.__qualname__}/v1")
          for cls in (ModelInitializationRequest, ModelReady)),
    ):
        data.register(codec)
    return data


def _source_policy(search, seed):
    # The old configured factory had no generation limit and raised at the
    # proposal cap. Do not borrow the corrected factory's future-run defaults.
    return BranchingPolicy(
        width=search.width, proposals_per_parent=search.proposals_per_parent,
        max_evaluations=search.max_evaluations, proposal_cap=search.proposal_cap,
        seed=seed, evaluation_seed=seed, proposal_seed=_generic_proposal_seed,
        proposal_seed_identity="astrolabe.configured-refinement.proposal-seed/v1",
        quota_mode="attempts", expansion="round_robin",
        proposal_batch_size=search.proposal_batch_size,
        evaluation_width=search.evaluation_width, pipeline=True,
        max_generations=None, cap_error=True,
    )


def _validate(source, output, connection):
    rows = {(kind, name): json.loads(value) for kind, name, value in
            connection.execute("SELECT kind,name,value FROM records")}
    data = _data(source / ".framework" / "data")
    verified = set()

    def verify(reference):
        if reference not in verified:
            value = data.get(reference)  # Native codec verifies the disk hash.
            verified.add(reference)
            for child in data._codecs[reference.codec].references(value):
                verify(child)
        return data.get(reference)

    def checkpoint(name):
        return verify(_blob(rows["checkpoint", name]))

    config_path = source / "config.yaml"
    config_sha256 = _sha256(config_path)
    provenance = json.loads((source / "provenance.json").read_text())
    cfg = OmegaConf.load(config_path)
    search = _search_settings(cfg)
    if (search.execution_mode != "isolated" or search.resume
            or cfg.search.get("max_generations") is not None):
        raise ValueError("Finalization supports the original isolated, uncapped-generation run only")
    error = {"type": "ValueError", "message": f"proposal cap {search.proposal_cap} would be exceeded"}
    if (provenance["state"] != "failed" or provenance["terminal"] is not True
            or provenance["error"] != error
            or Path(provenance["run"]["run_dir"]).resolve() != source
            or provenance["run"]["seed"] != cfg.seed
            or provenance["config"]["path"] != "config.yaml"
            or provenance["config"]["algorithm"] != "sha256"
            or provenance["config"]["sha256"] != config_sha256):
        raise ValueError("Source provenance/config does not prove a proposal-cap-failed run")
    scientific = key_for(("astrolabe.configured-refinement.scientific/v1",
                          _scientific_config_payload(cfg)))
    if rows["meta", "contract"]["revision"] != scientific.digest:
        raise ValueError("Source scientific configuration does not match its journal")
    assets = _read_asset_manifest(source / ".framework" / "assets.json")
    protocol = key_for(("astrolabe.configured-refinement.protocol/v3",
                        scientific, _asset_manifest_identity(assets)))
    prepared = checkpoint("configured_refinement/prepare")
    if prepared.protocol != protocol:
        raise ValueError("Source prepared checkpoint/config/assets binding changed")
    control_contract = checkpoint("control/contract")
    if not isinstance(control_contract, Key):
        raise ValueError("Missing source control contract")
    control = checkpoint("control/state")
    if control.request is not None or control.rng is None:
        raise ValueError("Source control checkpoint has a pending request or missing RNG evidence")
    state = verify(control.state)
    policy = _source_policy(search, int(cfg.seed))
    policy._validate_state(state)
    if (state.finished or state.stop_reason is not None or state.pending
            or state.queued or state.batch or state.refused
            or state.proposals != search.proposal_cap
            or state.next_evaluation != len(state.records)
            or len(state.records) >= search.max_evaluations):
        raise ValueError("Source must be unfinished, drained and at the proposal cap only")
    by_id = {member.id: member for member in state.records}
    if (set(by_id) != set(range(state.next_evaluation))
            or any(by_id.get(member.id) != member for member in (*state.frontier, *state.parents))):
        raise ValueError("Source frontier/parents do not match committed evaluation records")

    for (kind, name), row in rows.items():
        if kind == "attempt":
            if row["status"] != "completed" or row["result"] is None or row["rng"] is None:
                raise ValueError(f"Unresolved source attempt: {name}")
            for field in ("key", "inputs", "request", "result", "before", "rng"):
                verify(_blob(row[field]))
            for reference in row["references"]:
                verify(_blob(reference))
        if kind == "checkpoint" and name.startswith("control/issued/"):
            # Do not reconcile an issue lacking its original completion.
            checkpoint(name)
            if (kind, name.replace("control/issued/", "control/result/", 1)) not in rows:
                raise ValueError(f"Unresolved control issue: {name}")
            checkpoint(name.replace("control/issued/", "control/result/", 1))

    # Use only the worker's read validator, never its model/session initializer.
    worker = _Worker.__new__(_Worker)
    worker.root = source / ".prediction_workers"
    worker.data = data
    worker.codecs = data._codecs
    jobs, proofs = {}, {}
    for (kind, name), row in rows.items():
        if kind != "checkpoint" or not name.startswith("native-jobs/"):
            continue
        receipt = verify(_blob(row))
        if (receipt.status != "completed" or receipt.result is None
                or name != "native-jobs/" + receipt.identity):
            raise ValueError(f"Unresolved native job: {name}")
        task, result = verify(receipt.task), verify(receipt.result)
        if any(result.get(key) != value for key, value in task.items()):
            raise ValueError(f"Native result differs from its admitted task: {name}")
        job = verify(_worker_blob(task["job"]))
        if (job.identity != receipt.identity or task["kind"] != job.phase
                or task["id"] != job.id or task["seed"] != job.seed
                or task["generation"] != job.generation):
            raise ValueError(f"Native job/task identity mismatch: {name}")
        jobs[job.identity] = job
        if job.phase == "propose":
            proposal = verify(_worker_blob(result["proposal"]))
            if (proposal.id != job.id or job.parent is None
                    or proposal.parent_id != job.parent.id):
                raise ValueError(f"Native proposal identity mismatch: {name}")
            continue
        raw_path = worker.root / "receipts" / (task["_request_hash"] + ".json")
        score_path = raw_path.with_name(task["_request_hash"] + ".score.json")
        raw = worker._read(raw_path, task)
        scored = worker._read(score_path, task)
        if scored != result or any(scored.get(key) != value for key, value in raw.items()):
            raise ValueError(f"Worker raw/Score evidence differs from committed job: {name}")
        issued_path = raw_path.with_name(task["_request_hash"] + ".issued.json")
        if "root_prediction_import" not in raw or issued_path.exists():
            issued = json.loads(issued_path.read_text())
            if issued != dict(request=task, worker_pid=raw["worker_pid"]):
                raise ValueError(f"Worker issue does not match its completed prediction: {name}")
            proofs[str(issued_path)] = _sha256(issued_path)
        member = by_id[job.id]
        record = verify(member.value)
        candidate = verify(_worker_blob(task["candidate_blob"]))
        score = verify(_worker_blob(scored["score_receipt"]))
        proposal_ref = job.proposal.value if job.proposal is not None else job.source
        proposal = verify(proposal_ref)
        metrics = verify(record.metrics)
        if (task["scoring_identity"] != protocol.digest
                or score.scoring_identity != protocol
                or task["candidate_identity"] != candidate.identity.digest
                or task["system_identity"] != candidate.system.digest
                or record.candidate != candidate or proposal.candidate != candidate
                or proposal.protocol != protocol
                or record.generation != proposal.generation
                or record.parent_id != task["parent_id"]
                or record.proposal_step != task["step"]
                or record.label != task["sequence"] or member.key != record.label
                or member.score != record.score or record.score != score.measurement.score
                or record.scores != score.measured_scores
                or record.structure_path != score.measurement.structure_path
                or metrics != dict(score.measurement.metrics)):
            raise ValueError(f"Committed member/candidate/metrics do not match Score: {name}")
        # Metrics retain source paths. They must name actual verified structures,
        # not merely existing files detached from their Score evidence.
        structures = {item.assembly: item for item in verify(record.scores).structures}
        if set(structures) != set(metrics["structure_paths"]):
            raise ValueError(f"Score structures do not match committed metrics: {name}")
        for assembly, path in metrics["structure_paths"].items():
            path = Path(path)
            content = verify(structures[assembly].content)
            if not path.is_file() or path.read_bytes() != content:
                raise ValueError(f"Committed structure differs from native Score bytes: {path}")
            proofs[str(path)] = hashlib.sha256(content).hexdigest()
        if record.structure_path not in metrics["structure_paths"].values():
            raise ValueError(f"Primary structure is outside committed Score structures: {name}")
        proofs[str(raw_path)] = _sha256(raw_path)
        proofs[str(score_path)] = _sha256(score_path)
    if (set(jobs) != {f"propose/{i}" for i in range(1, state.proposals + 1)}
            | {f"evaluate/{i}" for i in by_id}):
        raise ValueError("Native job receipts do not cover the complete committed budget")
    expected_receipts = {Path(path) for path in proofs if Path(path).parent == worker.root / "receipts"}
    if set((worker.root / "receipts").glob("*.json")) != expected_receipts:
        raise ValueError("Worker receipt directory contains unmatched or unfinished evidence")
    for terminal in state.batch_results:
        if terminal.status != "completed" or jobs.get(terminal.job.identity) != terminal.job:
            raise ValueError("Final-generation proposal evidence is unfinished or unmatched")

    # A pure derived transition only: never call ask/drive, restore RNG, initialize
    # models, change cap_error, save a checkpoint, or claim source completion.
    derived = policy._advance_generation(state)
    if not derived.finished or derived.stop_reason != "proposal_cap" or not derived.frontier:
        raise ValueError("Native selection did not yield a proposal-cap final frontier")
    owner = NativeRefinementOwner.__new__(NativeRefinementOwner)
    owner.spec = SimpleNamespace(data=data)
    result = NativeRefinementResult(derived.frontier, derived, owner)
    selected_ids = {member.id for member in result.selected}
    evaluations = []
    for evaluation_id, record in result.member_items():
        metrics = result.metrics(record)
        evaluations.append(dict(
            evaluation_id=evaluation_id, generation=record.generation,
            parent_id=record.parent_id, score=record.score,
            candidate_identity=record.candidate.identity.digest,
            system_identity=record.candidate.system.digest,
            structure_paths=metrics["structure_paths"], loss=metrics["loss"],
            selected=evaluation_id in selected_ids,
        ))
    beam = []
    for member, record in zip(result.selected, result.members(), strict=True):
        metrics = result.metrics(record)
        beam.append(dict(evaluation_id=member.id, score=member.score,
                         candidate=_candidate_state(record.candidate),
                         structure_paths=metrics["structure_paths"], loss=metrics["loss"]))
    source_fields = dict(
        kind="derived_budget_snapshot", source_run=str(source), source_state="failed",
        source_state_finished=False, source_generation=state.generation,
        source_config_sha256=config_sha256,
        source_control_checkpoint_sha256=_blob(rows["checkpoint", "control/state"]).identity.digest,
        source_control_contract=control_contract.digest,
        source_control_policy=dict(cap_error=True, max_generations=None,
                                   selection=policy.selection_identity, archive_selection=False),
        stop_reason="proposal_cap", new_inference_count=0, new_score_count=0,
    )
    summary = dict(
        out_dir=str(output), routing=dict(algorithm="branching", owner="native_refinement"),
        seed=int(cfg.seed), repeats=1, finalization=source_fields,
        search=dict(
            width=search.width, proposals_per_parent=search.proposals_per_parent,
            max_evaluations=search.max_evaluations, max_generations=None,
            proposal_cap=search.proposal_cap, proposal_batch_size=search.proposal_batch_size,
            execution=OmegaConf.to_container(cfg.search.execution, resolve=True),
            evaluations=result.evaluations, proposals=result.proposals,
            duplicates=state.duplicates, rejected=state.rejected, generation=state.generation,
            stop_reason="proposal_cap", budget_complete=False,
        ), beam=beam, evaluations=evaluations,
    )
    for path in (config_path, source / "provenance.json", source / ".framework" / "assets.json",
                 source / ".framework" / "journal.sqlite"):
        proofs[str(path)] = _sha256(path)
    # Record the actual exporting code, separately from historical run provenance.
    tool_files = [Path(__file__), Path(__file__).parents[1] / "cli.py"]
    tool_files += [Path(__file__).with_name(name + ".py") for name in (
        "branching", "branch_types", "store", "native_refinement", "native_refinement_runtime",
        "native_scores", "refinement_worker", "configured_refinement", "control_journal")]
    receipt = dict(
        schema="astrolabe.refinement-finalization/v1", **source_fields,
        source_scientific_contract=scientific.digest, source_protocol=protocol.digest,
        source_files_sha256=proofs,
        verified_blobs=[asdict(ref) for ref in sorted(verified, key=lambda r: (r.codec, r.identity.digest))],
        tool_source_sha256={str(path): _sha256(path) for path in tool_files},
        evaluations=len(evaluations), selected_evaluation_ids=[item["evaluation_id"] for item in beam],
    )
    return summary, receipt


def finalize_run(run, output, *, dry_run=False):
    """Validate a frozen failed run and optionally publish a separate snapshot.

    Both overlap directions and all existing outputs are rejected. Dry-run uses
    the identical validators/selection and creates no directory or receipt.
    """
    source, destination = Path(run).expanduser().resolve(), Path(output).expanduser().resolve()
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("Finalization output must be outside, and not contain, the source run")
    if destination.exists():
        raise FileExistsError(f"Finalization output already exists: {destination}")
    try:
        # rb is intentional: lock an existing owner inode without creating or
        # writing it; this also works on a read-only bind on Linux.
        with (source / ".framework" / "owner.lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            connection = sqlite3.connect(
                (source / ".framework" / "journal.sqlite").as_uri() + "?mode=ro", uri=True)
            try:
                connection.execute("PRAGMA query_only=ON")
                connection.execute("BEGIN")
                summary, receipt = _validate(source, destination, connection)
            finally:
                connection.close()
            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(tempfile.mkdtemp(prefix=".finalize-", dir=destination.parent))
                try:
                    atomic_json(temporary / "03_final_beam.json", summary)
                    receipt["final_beam_sha256"] = _sha256(temporary / "03_final_beam.json")
                    atomic_json(temporary / "finalization_receipt.json", receipt)
                    if destination.exists():
                        raise FileExistsError(f"Finalization output already exists: {destination}")
                    temporary.rename(destination)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            return summary
    except BlockingIOError as error:
        raise ValueError("Source run still has a live owner") from error
    except (KeyError, TypeError, sqlite3.Error, ContractConflict, OutcomeUnknown) as error:
        raise ValueError(f"Incomplete or conflicting finalization evidence: {error}") from error
