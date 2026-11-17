"""CPU-only native receipt fixtures: no model, inference or Score execution."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
from omegaconf import OmegaConf

import astrolabe.cli as cli
from astrolabe.design.branch_owner import NativeJobReceipt, _RECEIPT
from astrolabe.design.branch_types import BranchJob, BranchMember, BranchProposal, BranchingState, branch_codec
from astrolabe.design.configured_refinement import (
    _PreparedCheckpoint, _PREPARED_CHECKPOINT_CODEC, _asset_manifest_identity,
    _asset_manifest_record, _candidate_label, _scientific_config_payload,
    _SelectedAssetManifest, _search_settings,
)
from astrolabe.design.control_journal import ControlCheckpoint, _CHECKPOINT
from astrolabe.design.native_refinement import (
    NativeMeasurement, RefinementMemberRecord, RefinementProposalRecord,
    _MEMBER_CODEC, _METRICS_CODEC, _PROPOSAL_CODEC,
)
from astrolabe.design.native_refinement_runtime import ScoreReceipt, _SCORE_RECEIPT_CODEC
from astrolabe.design.native_scores import snapshot_scores
from astrolabe.design.refinement_finalization import _data, _source_policy, finalize_run
from astrolabe.design.refinement_worker import _SCHEMA, _sidecars
from astrolabe.design.runtime import Runtime
from astrolabe.design.store import JSONCodec, design_codec
from astrolabe.design.types import Key, key_for
from astrolabe.evaluators.utils import StandardScores
from test_native_refinement_recovery import _case


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def frozen(tmp_path):
    # Reuse the native molecular fixture's candidate translation, not any model
    # callbacks or optimizer. All evidence below is explicitly synthetic.
    case = _case(tmp_path / "molecule")
    candidate = case.spec.initial_candidate
    case.close()
    source = tmp_path / "source"
    source.mkdir()
    cfg = OmegaConf.create(dict(
        seed=71, repeats=1, out_dir=str(source), steps=13,
        seqs={"A": "AC"}, types={"A": "protein"},
        evaluator={"evaluator_terms": {"fold": {"_type": "dummy"}}},
        mutator={"mutator_terms": {"mutate": {"_type": "dummy"}}},
        stages={"default": {"oligo": ["A"], "evaluator": "fold", "mutators": ["mutate"]}},
        search=dict(algorithm="branching", owner="native_refinement", measurement="weighted_score",
                    width=2, proposals_per_parent=2, max_evaluations=10, proposal_cap=2,
                    execution=dict(mode="isolated", workers=2, in_flight=2)),
    ))
    OmegaConf.save(cfg, source / "config.yaml")
    scientific = key_for(("astrolabe.configured-refinement.scientific/v1", _scientific_config_payload(cfg)))
    assets = _SelectedAssetManifest((), ())
    _write(source / ".framework/assets.json", _asset_manifest_record(assets))
    protocol = key_for(("astrolabe.configured-refinement.protocol/v3", scientific,
                        _asset_manifest_identity(assets)))
    data = _data(source / ".framework/data")
    runtime = Runtime(root=source / ".framework", data=data, scientific_contract=scientific)
    rng = data.put({"synthetic_rng": True}, design_codec)
    candidate_ref = data.put(candidate, design_codec)
    runtime.save("configured_refinement/prepare", _PreparedCheckpoint(protocol, candidate_ref, rng, rng),
                 _PREPARED_CHECKPOINT_CODEC)
    runtime.save("control/contract", key_for("original-control-contract"))
    members = []
    for identifier in range(3):
        proposal = RefinementProposalRecord(candidate, _candidate_label(candidate),
                                            0 if identifier == 0 else 1,
                                            -1 if identifier == 0 else 0,
                                            identifier, identifier, 71, protocol)
        proposal_ref = data.put(proposal, _PROPOSAL_CODEC)
        job = BranchJob("evaluate", identifier, proposal.generation, 71, source=proposal_ref)
        job_ref = data.put(job, branch_codec(BranchJob))
        task = dict(kind="evaluate", id=identifier, generation=job.generation, seed=71,
                    parent_id=None if identifier == 0 else 0, step=identifier,
                    job=asdict(job_ref), candidate_blob=asdict(candidate_ref),
                    candidate_identity=candidate.identity.digest, system_identity=candidate.system.digest,
                    sequence=proposal.label, scoring_identity=protocol.digest)
        task["_request_hash"] = key_for(task).digest
        root = source / ".prediction_workers"
        raw_dir = root / "requests" / task["_request_hash"] / "raw_model_results"
        score_dir = root / "measurements" / task["_request_hash"] / "raw_model_results"
        raw_dir.mkdir(parents=True)
        score_dir.mkdir(parents=True)
        structure = score_dir.parent / "structures" / "A.cif"
        structure.parent.mkdir()
        structure.write_text(f"data_fixture_{identifier}\n")
        raw_scores = snapshot_scores(StandardScores(names=["A"], plddt={"A": [0.5]},
                                                    complex_paths={"A": structure}), data)
        metrics = dict(structure_paths={"A": str(structure)}, loss={"raw": [identifier, 0.5]})
        measurement = NativeMeasurement(1.0, metrics, str(structure))  # All tied: ID order matters.
        score_ref = data.put(ScoreReceipt(Key(task["_request_hash"]), raw_scores, protocol,
                                         raw_scores, measurement, rng), _SCORE_RECEIPT_CODEC)
        data.persist((score_ref, job_ref, proposal_ref))
        raw = dict(**task, schema=_SCHEMA, request=task, raw_scores=asdict(raw_scores),
                   prediction_after_rng=asdict(rng), worker_pid=100, constructor_count=1,
                   prediction_wall_s=0.0, job_raw_dir=str(raw_dir), sidecars=_sidecars(raw_dir),
                   timing=dict(initialization_seconds=0.0, evaluation_seconds=0.0,
                               persistence_seconds=0.0, total_seconds=0.0))
        scored = dict(**raw, score_receipt=asdict(score_ref), score_worker_pid=101,
                      score_raw_dir=str(score_dir), score_sidecars=_sidecars(score_dir))
        _write(root / "receipts" / (task["_request_hash"] + ".json"), raw)
        _write(root / "receipts" / (task["_request_hash"] + ".issued.json"),
               dict(request=task, worker_pid=100))
        _write(root / "receipts" / (task["_request_hash"] + ".score.json"), scored)
        runtime.save("native-jobs/" + job.identity, NativeJobReceipt(
            job.identity, data.put(task, JSONCodec()), "completed", data.put(scored, JSONCodec())), _RECEIPT)
        record = RefinementMemberRecord(candidate, raw_scores, 1.0, str(structure),
                                         data.put(metrics, _METRICS_CODEC), proposal.label,
                                         proposal.generation, task["parent_id"], identifier)
        member = BranchMember(identifier, proposal.label, data.put(record, _MEMBER_CODEC), 1.0)
        members.append(member)
    for identifier in (1, 2):
        job = BranchJob("propose", identifier, 1, 71, parent=members[0], source=members[0].value)
        job_ref = data.put(job, branch_codec(BranchJob))
        proposal = BranchProposal(identifier, 0, members[identifier].key,
                                  data.put(RefinementProposalRecord(candidate, members[identifier].key,
                                           1, 0, identifier, identifier, 71, protocol), _PROPOSAL_CODEC))
        task = dict(kind="propose", id=identifier, generation=1, seed=71, parent_id=0,
                    job=asdict(job_ref))
        task["_request_hash"] = key_for(task).digest
        proposal_ref = data.put(proposal, branch_codec(BranchProposal))
        data.persist((proposal_ref, job_ref))
        result = dict(**task, proposal=asdict(proposal_ref),
                      after_rng=asdict(rng))
        runtime.save("native-jobs/" + job.identity, NativeJobReceipt(
            job.identity, data.put(task, JSONCodec()), "completed", data.put(result, JSONCodec())), _RECEIPT)
    state = BranchingState(generation=1, proposals=2, next_evaluation=3, records=tuple(members),
                           frontier=(members[2], members[1]), parents=(members[0],))
    runtime.save("control/state", ControlCheckpoint(data.put(state, branch_codec(BranchingState)), rng=rng),
                 _CHECKPOINT)
    data.persist((candidate_ref, rng))
    runtime.close()
    _write(source / "provenance.json", dict(
        state="failed", terminal=True,
        error=dict(type="ValueError", message="proposal cap 2 would be exceeded"),
        run=dict(run_dir=str(source), seed=71),
        config=dict(path="config.yaml", algorithm="sha256",
                    sha256=hashlib.sha256((source / "config.yaml").read_bytes()).hexdigest()),
    ))
    return source, state, cfg


def test_read_only_export_uses_native_frontier_ties_and_preserves_source(frozen, tmp_path, monkeypatch):
    source, state, cfg = frozen
    before = _hashes(source)
    monkeypatch.setattr(cli, "_compose_hydra_config", lambda **kw: pytest.fail("Hydra must not run"))
    from astrolabe.design.refinement_worker import _Worker
    monkeypatch.setattr(_Worker, "__init__", lambda *a, **kw: pytest.fail("No worker initializer"))
    output = tmp_path / "export"
    parser = cli._build_parser()
    argv = ["run", "--finalize-run", str(source), "--finalize-output", str(output)]
    ns = parser.parse_args([*argv, "--dry-run"])
    assert ns._handler(ns) == 0
    assert not output.exists()
    ns = parser.parse_args(argv)
    assert ns._handler(ns) == 0
    summary = json.loads((output / "03_final_beam.json").read_text())
    expected = _source_policy(_search_settings(cfg), 71)._advance_generation(state).frontier
    assert [row["evaluation_id"] for row in summary["beam"]] == [row.id for row in expected] == [1, 2]
    assert len(summary["evaluations"]) == 3  # Root is retained, not selected from the archive.
    assert summary["finalization"]["source_state"] == "failed"
    assert summary["finalization"]["source_state_finished"] is False
    assert summary["finalization"]["source_control_policy"]["max_generations"] is None
    assert summary["finalization"]["new_inference_count"] == summary["finalization"]["new_score_count"] == 0
    receipt = json.loads((output / "finalization_receipt.json").read_text())
    assert receipt["final_beam_sha256"] == hashlib.sha256((output / "03_final_beam.json").read_bytes()).hexdigest()
    assert _hashes(source) == before
    with pytest.raises(FileExistsError):
        finalize_run(source, output)


@pytest.mark.parametrize("damage", ["pending", "unknown", "corrupt", "config", "score", "sidecar", "missing"])
def test_invalid_evidence_fails_before_any_output(frozen, tmp_path, damage):
    source, state, cfg = frozen
    data = _data(source / ".framework/data")
    if damage == "pending":
        runtime = Runtime(root=source / ".framework", data=data,
                          scientific_contract=key_for(("astrolabe.configured-refinement.scientific/v1",
                                                       _scientific_config_payload(cfg))))
        control = runtime.load("control/state", _CHECKPOINT)
        runtime.save("control/state", replace(control, request=data.put({"pending": True}, design_codec)), _CHECKPOINT)
        runtime.close()
    elif damage == "unknown":
        with sqlite3.connect(source / ".framework/journal.sqlite") as db:
            db.execute("INSERT INTO records VALUES (?,?,?)", ("attempt", "uncertain", json.dumps(dict(status="unknown"))))
    elif damage == "corrupt":
        data._path_for(state.records[1].value).write_text("corrupt bytes")
    elif damage == "config":
        cfg.seed = 72
        OmegaConf.save(cfg, source / "config.yaml")
    elif damage == "score":
        path = next((source / ".prediction_workers/receipts").glob("*.score.json"))
        value = json.loads(path.read_text())
        value["score_receipt"]["identity"]["digest"] = "0" * 64
        _write(path, value)
    elif damage == "sidecar":
        path = next((source / ".prediction_workers/requests").glob("*/raw_model_results"))
        (path / "late.txt").write_text("uncommitted output")
    else:
        next((source / ".prediction_workers/receipts").glob("*.score.json")).unlink()
    before = _hashes(source)
    output = tmp_path / "absent-parent" / "export"
    with pytest.raises((ValueError, FileNotFoundError)):
        finalize_run(source, output)
    assert not output.parent.exists()
    assert _hashes(source) == before


@pytest.mark.parametrize("extra", [["--task", "anything"], ["--standalone", "config.yaml"],
                                   ["--overrides", "steps=1"], ["steps=1"], ["--print-config"]])
def test_cli_rejects_config_mixing_before_reading_source(extra):
    parser = cli._build_parser()
    ns = parser.parse_args(["run", "--finalize-run", "/missing", "--finalize-output", "/elsewhere", *extra])
    with pytest.raises(ValueError, match="cannot be mixed"):
        ns._handler(ns)


def test_overlap_and_live_owner_fail_closed(frozen):
    import fcntl
    source, _, _ = frozen
    for output in (source, source / "derived", source.parent):
        with pytest.raises(ValueError, match="outside"):
            finalize_run(source, output)
    with (source / ".framework/owner.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="live owner"):
            finalize_run(source, source.parent / "locked-export")
