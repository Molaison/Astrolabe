"""Model-free integrity validation for a completed native protein beam."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def _blob(record):
    from astrolabe.design.types import Blob, Key

    if not isinstance(record, dict) or not isinstance(record.get("identity"), str) or not isinstance(record.get("codec"), str):
        raise ValueError("Journal contains an invalid blob reference")
    return Blob(Key(record["identity"]), record["codec"])


def _contract_digest(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def _read_journal(path):
    if not path.is_file():
        raise FileNotFoundError(f"Completed beam control journal is missing: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)]:
            raise ValueError("Completed beam control journal failed SQLite integrity_check")
        records = {}
        for kind, name, value in connection.execute("SELECT kind,name,value FROM records"):
            key = (kind, name)
            if key in records:
                raise ValueError(f"Completed beam control journal has duplicate record {key}")
            decoded = json.loads(value)
            if not isinstance(decoded, dict):
                raise ValueError(f"Completed beam control journal record is not an object: {key}")
            records[key] = decoded
        return records
    finally:
        connection.close()


def _verify_scientific_identity(state, cfg):
    from astrolabe.protein_beam import _scientific_contract_config
    from astrolabe.protein_beam_contract import scientific_contract

    expected, digest = scientific_contract(_scientific_contract_config(cfg))
    stored = state.get("scientific_contract")
    if not isinstance(stored, dict):
        raise ValueError("Completed beam checkpoint has no scientific contract")
    if _contract_digest(stored) != state.get("config_hash"):
        raise ValueError("Completed beam checkpoint scientific contract digest changed")
    if state.get("config_hash") != digest or stored != expected:
        raise ValueError("Scientific config, data, weights or implementation changed on resume")
    return expected, digest


def _load_control_journal(output, state):
    from astrolabe.design.branch_owner import _RECEIPT
    from astrolabe.design.branch_types import (
        BranchAction,
        BranchEvent,
        BranchingState,
        branch_codec,
    )
    from astrolabe.design.control_journal import _CHECKPOINT
    from astrolabe.design.store import DataStore, JSONCodec, design_codec
    framework = Path(output) / ".framework"
    journal = _read_journal(framework / "journal.sqlite")
    meta = journal.get(("meta", "contract"))
    if not isinstance(meta, dict):
        raise ValueError("Completed beam control journal has no contract metadata")
    if (
        meta.get("journal_version") != 1
        or meta.get("revision") != state.get("config_hash")
        or meta.get("operations") != {}
        or meta.get("budgets") != {}
    ):
        raise ValueError("Completed beam control journal belongs to another execution")

    data = DataStore(framework / "data")
    record_codec = JSONCodec("astrolabe.native-record/v1")
    job_codec = JSONCodec()
    state_codec = branch_codec(BranchingState)
    for codec in (
        design_codec,
        record_codec,
        job_codec,
        state_codec,
        branch_codec(BranchAction),
        branch_codec(BranchEvent),
        _RECEIPT,
        _CHECKPOINT,
    ):
        data.register(codec)

    control_checkpoint_record = journal.get(("checkpoint", "control/state"))
    if control_checkpoint_record is None:
        raise ValueError("Completed beam control journal has no final control checkpoint")
    checkpoint = data.get(_blob(control_checkpoint_record), _CHECKPOINT)
    if checkpoint.request is not None:
        raise ValueError("Completed beam control journal retains a pending control request")
    if checkpoint.rng is None:
        raise ValueError("Completed beam control journal has no RNG receipt")
    rng = data.get(checkpoint.rng, design_codec)
    if not isinstance(rng, dict) or set(rng) != {"numpy", "python"}:
        raise ValueError("Completed beam control journal RNG receipt belongs to another execution")
    control = data.get(checkpoint.state, state_codec)
    if (
        not control.finished
        or control.pending
        or control.queued
        or control.batch
    ):
        raise ValueError("Completed beam control journal is not a finished terminal state")

    issued = {
        name.removeprefix("control/issued/")
        for kind, name in journal
        if kind == "checkpoint" and name.startswith("control/issued/")
    }
    completed = {
        name.removeprefix("control/result/")
        for kind, name in journal
        if kind == "checkpoint" and name.startswith("control/result/")
    }
    if issued - completed:
        raise ValueError("Completed beam control journal has issued requests without terminal results")

    receipt_records = {
        name: record
        for (kind, name), record in journal.items()
        if kind == "checkpoint" and name.startswith("native-jobs/")
    }
    native_receipts = {}
    for name, record in receipt_records.items():
        receipt = data.get(_blob(record), _RECEIPT)
        if receipt.status in {"reserved", "issued", "unknown"}:
            raise ValueError(f"Completed beam control journal has unresolved native job: {name}")
        native_receipts[name] = receipt
    return data, record_codec, job_codec, control, native_receipts


def _verify_native_record(record, output):
    from astrolabe.protein_beam_contract import verify_native_artifacts

    if record.get("status") != "done":
        raise ValueError(f"Completed beam contains a nonterminal candidate {record.get('id')}")
    artifacts = record.get("native_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"snapshot", "structure_path"}:
        raise ValueError(f"Candidate {record.get('id')} has no complete native artifact manifest")
    for name in ("snapshot", "structure_path"):
        specification = artifacts[name]
        if Path(specification["path"]).resolve() != Path(record[name]).resolve():
            raise ValueError(f"Candidate {record.get('id')} {name} identity changed")
    verify_native_artifacts(artifacts)
    return artifacts


def _verify_native_jobs(state, output, data, job_codec, receipts):
    from astrolabe.protein_beam_calibration import validate_native_result

    records = {int(record["id"]): record for record in state["records"]}
    evaluations = {}
    for name, receipt in receipts.items():
        if not name.startswith("native-jobs/evaluate/"):
            continue
        if receipt.status != "completed" or receipt.result is None:
            raise ValueError(f"Completed beam evaluation receipt is not successful: {name}")
        task = data.get(receipt.task, job_codec)
        raw = data.get(receipt.result, job_codec)
        candidate_id = int(task["id"])
        if candidate_id in evaluations:
            raise ValueError(f"Completed beam has duplicate evaluation receipt for candidate {candidate_id}")
        if candidate_id not in records:
            raise ValueError(f"Evaluation receipt has no completed beam record: {candidate_id}")
        validate_native_result(task, raw, output)
        record = records[candidate_id]
        if any(record.get(key) != value for key, value in raw.items()):
            raise ValueError(f"Checkpoint candidate {candidate_id} differs from its native receipt")
        terminal = Path(output) / "candidates" / f"{candidate_id:06d}" / "result.json"
        if not terminal.is_file():
            raise FileNotFoundError(f"Candidate terminal result is missing: {terminal}")
        terminal_value = json.loads(terminal.read_text())
        validate_native_result(task, terminal_value, output)
        if terminal_value != raw:
            raise ValueError(f"Candidate terminal result changed: {terminal}")
        _verify_native_record(raw, output)
        evaluations[candidate_id] = raw
    expected = {candidate_id for candidate_id, record in records.items() if record.get("status") == "done"}
    if set(evaluations) != expected:
        raise ValueError(
            "Completed beam evaluation receipts do not cover exactly its terminal candidates"
        )


def _verify_control_projection(state, data, record_codec, control):
    records = {}
    for member in control.records:
        value = data.get(member.value, record_codec)
        if int(value.get("id")) != member.id:
            raise ValueError("Control journal member id differs from its native record")
        if member.id in records:
            raise ValueError("Control journal contains duplicate candidate ids")
        records[member.id] = value
    native = {int(record["id"]): record for record in state["records"]}
    if set(records) != set(native):
        raise ValueError("Beam checkpoint candidates differ from the control journal")
    for candidate_id, record in records.items():
        if any(native[candidate_id].get(key) != value for key, value in record.items()):
            raise ValueError("Beam checkpoint candidates differ from the control journal")
        if set(native[candidate_id]) - set(record) - {"selection"}:
            raise ValueError("Beam checkpoint added an unverified candidate field")
    if [member.id for member in control.frontier] != [int(record["id"]) for record in state["beam"]]:
        raise ValueError("Beam checkpoint frontier differs from the control journal")
    expected = {
        "generation": control.generation,
        "proposals": control.proposals,
        "rejected_proposals": control.rejected,
        "duplicate_proposals": control.duplicates,
        "stop_reason": control.stop_reason,
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("Beam checkpoint terminal counters differ from the control journal")


def _verify_paid_ledger(cfg, output, state):
    policy = cfg.protein_beam.get("af3_call_budget")
    if not policy or not policy.get("ledger"):
        return
    path = Path(str(policy["ledger"]))
    if not path.is_file():
        raise FileNotFoundError(f"Native call ledger is missing: {path}")
    ledger = json.loads(path.read_text())
    expected_contract = {
        "max_total": int(policy["max_total"]),
        "max_per_case": int(policy["max_per_case"]),
    }
    if ledger.get("contract") != expected_contract or not isinstance(ledger.get("entries"), dict):
        raise ValueError("Native call ledger contract changed")
    for record in state["records"]:
        token = record.get("af3_call_token")
        if token is None:
            continue
        entry = ledger["entries"].get(token)
        if not isinstance(entry, dict) or entry.get("status") != "issued":
            raise ValueError(f"Paid native call token is not durably issued: {token}")


def _verify_root_context(cfg, output, state):
    from astrolabe.protein_beam_calibration import (
        calibration_enabled,
        calibrate_root,
        verify_calibrated_state,
    )

    settings = cfg.protein_beam
    if calibration_enabled(cfg):
        root = verify_calibrated_state(cfg, output, state)
        if root is None:
            raise ValueError("Completed calibrated beam has no completed native root")
        evidence = calibrate_root(cfg, output, state, root)
        saved = Path(output) / "root_calibration.json"
        if not saved.is_file() or json.loads(saved.read_text()) != evidence:
            raise ValueError("Completed native root calibration artifact changed")
        return
    reference = settings.get("root_parity_reference")
    if not reference:
        return
    from astrolabe.protein_beam_root_parity import root_parity

    root = next(
        (record for record in state["records"] if int(record["id"]) == 0 and record.get("status") == "done"),
        None,
    )
    if root is None:
        raise ValueError("Completed root-parity beam has no completed native root")
    evidence = root_parity(root, str(reference))
    saved = Path(output) / "root_parity.json"
    if not evidence["passed"] or not saved.is_file() or json.loads(saved.read_text()) != evidence:
        raise ValueError("Completed native root parity evidence changed")


def _verify_exports(cfg, output, state):
    from astrolabe.protein_beam_policy import pareto_select

    settings = cfg.protein_beam
    holes_ranking = str(settings.get("holes_ranking", "raw_sum"))
    expected_final = {
        key: state[key]
        for key in ("beam", "status", "budget_complete", "wall_s", "config_hash")
    }
    final_path = Path(output) / "final_beam.json"
    if not final_path.is_file() or json.loads(final_path.read_text()) != expected_final:
        raise ValueError("Completed final_beam.json differs from the checkpoint")
    if str(settings.get("selection", "weighted")) == "pareto":
        archive = pareto_select(copy.deepcopy(state["records"]), len(state["records"]), holes_ranking)
        expected_archive = [record for record in archive if record["selection"]["pareto_front"] == 0]
        archive_path = Path(output) / "pareto_archive.json"
        if not archive_path.is_file() or json.loads(archive_path.read_text()) != expected_archive:
            raise ValueError("Completed pareto_archive.json differs from the checkpoint")
    scores = Path(output) / "scores.csv"
    if not scores.is_file():
        raise FileNotFoundError(f"Completed score table is missing: {scores}")
    with scores.open(newline="") as handle:
        rows = {int(row["id"]): row for row in csv.DictReader(handle)}
    if set(rows) != {int(record["id"]) for record in state["records"]}:
        raise ValueError("Completed score table does not cover exactly its candidates")
    for record in state["records"]:
        row = rows[int(record["id"])]
        if row["sequence"] != record["sequence"] or row["structure_path"] != record["structure_path"]:
            raise ValueError(f"Completed score row changed for candidate {record['id']}")
        if float(row["score"]) != float(record["score"]):
            raise ValueError(f"Completed score value changed for candidate {record['id']}")
        if any(float(row[name]) != float(record["metrics"][name]) for name in record["metrics"]):
            raise ValueError(f"Completed score metrics changed for candidate {record['id']}")


def validate_completed_beam(
    cfg,
    output,
    state=None,
    *,
    require_exports=True,
):
    """Validate a completed beam without loading a model or changing its files."""
    from astrolabe.protein_beam import select_beam, validate_config
    from astrolabe.protein_beam_policy import candidate_rejections, pareto_select

    validate_config(cfg)
    output = Path(output).expanduser().resolve()
    checkpoint = output / "beam_state.json"
    if state is None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Completed beam checkpoint is missing: {checkpoint}")
        state = json.loads(checkpoint.read_text())
    if not isinstance(state, dict) or state.get("status") != "complete":
        raise ValueError("Beam recovery preflight requires a completed checkpoint")
    if not isinstance(state.get("records"), list) or not isinstance(state.get("pending"), list):
        raise ValueError("Completed beam checkpoint has no candidate/pending lists")
    if state["pending"]:
        raise ValueError("Completed beam checkpoint still has pending work")
    ids = [record.get("id") for record in state["records"]]
    if not ids or len(ids) != len(set(ids)) or any(type(value) is not int or value < 0 for value in ids):
        from astrolabe.protein_beam_calibration import calibration_enabled
        if calibration_enabled(cfg):
            raise ValueError(
                "Completed calibrated beam requires exactly one native root; "
                "checkpoint has invalid or duplicate candidate ids"
            )
        raise ValueError("Completed beam checkpoint has invalid or duplicate candidate ids")
    if state.get("budget_complete") != (len(state["records"]) == int(cfg.protein_beam.max_evaluations)):
        raise ValueError("Completed beam budget flag differs from its terminal records")
    if not all(math.isfinite(float(state.get(key))) for key in ("started_at", "completed_at", "wall_s")):
        raise ValueError("Completed beam has invalid timing metadata")

    _verify_scientific_identity(state, cfg)
    data, record_codec, job_codec, control, receipts = _load_control_journal(output, state)
    _verify_native_jobs(state, output, data, job_codec, receipts)
    for record in state["records"]:
        _verify_native_record(record, output)
    _verify_paid_ledger(cfg, output, state)
    _verify_root_context(cfg, output, state)

    expected_records = copy.deepcopy(state["records"])
    root = next(
        (record for record in expected_records if int(record["id"]) == 0 and record.get("status") == "done"),
        None,
    )
    if root is None:
        raise ValueError("Completed beam has no completed native root")
    gates = state.get("calibrated_gates") or cfg.protein_beam.get("gates") or {}
    for record in expected_records:
        if record.get("status") != "done":
            continue
        reasons = candidate_rejections(record, root["metrics"], gates)
        expected = dict(record, eligible=not reasons, rejection_reasons=reasons)
        if record != expected:
            raise ValueError(f"Candidate {record['id']} eligibility differs from its metrics")
    holes_ranking = str(cfg.protein_beam.get("holes_ranking", "raw_sum"))
    if str(cfg.protein_beam.get("selection", "weighted")) == "pareto":
        selected = pareto_select(copy.deepcopy(expected_records), int(cfg.protein_beam.width), holes_ranking)
    else:
        selected = select_beam(expected_records, int(cfg.protein_beam.width))
    if [record["id"] for record in selected] != [record["id"] for record in state["beam"]]:
        raise ValueError("Completed beam frontier cannot be reproduced from its records")
    state["beam"] = selected
    _verify_control_projection(state, data, record_codec, control)
    if require_exports:
        _verify_exports(cfg, output, state)
    return state
