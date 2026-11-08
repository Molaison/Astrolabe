"""CPU-only posthoc state/publication fault injection (O11/O12, F23)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

import astrolabe.posthoc.collect as collect
from astrolabe.posthoc import db, filtering, publication
from astrolabe.posthoc.derive import derive_boltz


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    run = root / "task" / "001_run"
    run.mkdir(parents=True)
    (run / "scores.csv").write_text("step,weighted_score\n0,1.0\n")
    calls = {}
    for name in ("af3", "protenix", "boltz", "chai1", "channel_hole"):
        def derive(*args, _name=name, **kwargs):
            calls[_name] = calls.get(_name, 0) + 1
            return {(0, "AB"): {_name + "_metric": calls[_name]}}
        monkeypatch.setattr(collect, "derive_" + name, derive)
    arguments = dict(out_root=root, db_path=tmp_path / "scores.sqlite", workers=1)
    return run, arguments, calls


def receipt(arguments, run):
    conn = db.open_db(arguments["db_path"])
    try:
        return db.get_collection_receipt(conn, str(run))
    finally:
        conn.close()


def test_f23_partial_resume_retries_only_failed_derivation(source, monkeypatch):
    run, arguments, calls = source
    # Existing refold evidence must never be rewritten or used to request inference.
    refold = run / "refold_receipt.json"
    refold.write_text('{"status":"COMPLETE","prediction":"retained"}')
    original = refold.read_bytes()
    good = collect.derive_boltz
    def failed(*args, **kwargs):
        calls["boltz"] = calls.get("boltz", 0) + 1
        raise RuntimeError("F23 deterministic derivation failure")
    monkeypatch.setattr(collect, "derive_boltz", failed)
    first = collect.collect_scores(**arguments)
    assert first["status"] == "PARTIAL"
    proof = receipt(arguments, run)
    assert proof["status"] == "PARTIAL"
    assert proof["derivations"]["boltz"]["status"] == "FAILED"
    assert "F23" in proof["derivations"]["boltz"]["outcome"]["error"]
    monkeypatch.setattr(collect, "derive_boltz", good)
    second = collect.collect_scores(**arguments)
    assert second["runs_skipped"] == 0
    assert second["status"] == "COMPLETE"
    assert calls == {"af3": 1, "protenix": 1, "boltz": 2, "chai1": 1, "channel_hole": 1}
    assert receipt(arguments, run)["derivations"]["af3"] == proof["derivations"]["af3"]
    assert collect.collect_scores(**arguments)["runs_skipped"] == 1
    assert refold.read_bytes() == original
    conn = db.open_db(arguments["db_path"])
    try:
        metrics = json.loads(next(db.iter_records(conn))["metrics_json"])
        assert metrics["af3_metric"] == 1 and metrics["boltz_metric"] == 2
    finally:
        conn.close()


@pytest.mark.parametrize("dependency", ["scores.csv", "structures/step_000_AB_model.cif",
                                      "raw_model_results/af3/input.json", "config.yaml"])
def test_same_mtime_size_changed_content_cannot_skip(source, dependency):
    run, arguments, calls = source
    path = run / dependency
    path.parent.mkdir(parents=True, exist_ok=True)
    if dependency != "scores.csv":
        path.write_text("{}\n" if dependency.endswith("yaml") else "abc")
    collect.collect_scores(**arguments)
    stat = path.stat()
    path.write_bytes(path.read_bytes().replace(b"1.0", b"2.0") if dependency == "scores.csv"
                     else (b"{} " if dependency.endswith("yaml") else b"def"))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    assert collect.collect_scores(**arguments)["runs_processed"] == 1
    assert all(value == 2 for value in calls.values())


def test_external_source_config_and_derivation_version_are_identity(source, monkeypatch, tmp_path):
    run, arguments, _ = source
    original = tmp_path / "original"
    original.mkdir()
    config = original / "config.yaml"
    config.write_text("{}\n")
    (run / "scores.csv").write_text(f"step,src_run_dir\n0,{original}\n")
    collect.collect_scores(**arguments)
    stat = config.stat()
    config.write_text("{} ")
    os.utime(config, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert collect.collect_scores(**arguments)["runs_processed"] == 1
    monkeypatch.setattr(collect, "_COLLECTION_DERIVATION_VERSION", 999)
    assert collect.collect_scores(**arguments)["runs_processed"] == 1


def test_symlinked_directory_dependency_content_is_hashed(source, tmp_path):
    run, arguments, _ = source
    external = tmp_path / "external_raw"
    external.mkdir()
    raw = external / "payload.json"
    raw.write_text("abc")
    (run / "raw_model_results").mkdir()
    (run / "raw_model_results" / "boltz").symlink_to(external, target_is_directory=True)
    collect.collect_scores(**arguments)
    stat = raw.stat()
    raw.write_text("def")
    os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert collect.collect_scores(**arguments)["runs_processed"] == 1


def test_required_contract_is_not_a_legacy_signature(source):
    run, arguments, calls = source
    collect.collect_scores(**arguments)
    with pytest.raises(collect.MissingRequiredMetricsError):
        collect.collect_scores(**arguments, required_metrics=["not_provided"])
    assert receipt(arguments, run)["status"] == "PARTIAL"
    before = dict(calls)
    with pytest.raises(collect.MissingRequiredMetricsError):
        collect.collect_scores(**arguments, required_metrics=["not_provided"])
    assert calls == before


def test_real_parser_failure_remains_partial_and_preserves_independent_success(source, monkeypatch):
    run, arguments, calls = source
    raw = run / "raw_model_results/boltz/step_000_AB/predictions/AB/AB_model_0.json"
    raw.parent.mkdir(parents=True)
    raw.write_text("not-json")
    monkeypatch.setattr(collect, "derive_boltz", derive_boltz)
    assert collect.collect_scores(**arguments)["status"] == "PARTIAL"
    assert collect.collect_scores(**arguments)["runs_skipped"] == 0
    assert calls["af3"] == 1
    assert receipt(arguments, run)["derivations"]["boltz"]["status"] == "FAILED"


def test_additive_v2_migration_cannot_skip_legacy_signature(source):
    run, arguments, _ = source
    conn = sqlite3.connect(arguments["db_path"])
    conn.executescript("""
        CREATE TABLE runs (run_dir TEXT PRIMARY KEY, task TEXT NOT NULL,
            run TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}',
            scores_mtime_ns INTEGER NOT NULL, scores_size INTEGER NOT NULL, updated_at REAL NOT NULL);
    """)
    stat = (run / "scores.csv").stat()
    metadata = {"posthoc_collection_context": collect._collection_context(binder_chain="B", target_chain="A")}
    conn.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, 0)",
                 (str(run), "task", "001_run", json.dumps(metadata), stat.st_mtime_ns, stat.st_size))
    conn.commit()
    conn.close()
    assert collect.collect_scores(**arguments)["runs_processed"] == 1
    assert receipt(arguments, run)["status"] == "COMPLETE"


def test_f23_process_death_after_records_rolls_back_status_and_receipts(source):
    run, arguments, _ = source
    collect.collect_scores(**arguments)
    before = receipt(arguments, run)
    command = """
import os, sys
from astrolabe.posthoc import db
conn = db.open_db(sys.argv[1])
def crash(*args, **kwargs):
    os._exit(23)
db.upsert_run_signature = crash
db.commit_collection(conn, run_dir=sys.argv[2], task='task', run='001_run',
    metadata={}, scores_mtime_ns=0, scores_size=0, records=[],
    input_identity='should-not-commit', derivations={}, complete=False)
"""
    result = subprocess.run([sys.executable, "-c", command, str(arguments["db_path"]), str(run)])
    assert result.returncode == 23
    assert receipt(arguments, run) == before
    conn = db.open_db(arguments["db_path"])
    try:
        assert len(list(db.iter_records(conn))) == 1
    finally:
        conn.close()
    assert collect.collect_scores(**arguments)["runs_skipped"] == 1


def test_changed_input_replaces_obsolete_rows(source):
    run, arguments, _ = source
    collect.collect_scores(**arguments)
    (run / "scores.csv").write_text("step,weighted_score\n1,1.0\n")
    collect.collect_scores(**arguments)
    conn = db.open_db(arguments["db_path"])
    try:
        assert [row["step"] for row in db.iter_records(conn)] == [1]
    finally:
        conn.close()


@pytest.fixture
def filter_input(tmp_path):
    source = tmp_path / "input.sqlite"
    conn = db.open_db(source)
    records = []
    for step in range(2):
        structure = tmp_path / f"structure{step}.cif"
        structure.write_text(f"data_{step}\n")
        records.append(dict(run_dir=str(tmp_path / "run"), task="task", run="001_run",
                            step=step, oligo="AB", structure_path=str(structure),
                            metrics={"weighted_score": 1.0}, scores_mtime_ns=1, scores_size=1))
    db.upsert_records(conn, records)
    conn.close()
    return dict(db_path=source, dst_dir=tmp_path / "filtered", filter_expr="weighted_score > 0")


def manifest(arguments):
    return json.loads((arguments["dst_dir"] / "bundle_manifest.json").read_text())


def assert_bundle(proof):
    assert proof["status"] == "COMPLETE"
    for item in proof["outputs"]:
        assert publication.content_digest(Path(item["path"])) == item["sha256"]


def test_filter_complete_bundle_and_failed_metric_gate_preserve_previous(filter_input):
    result = filtering.filter_structures(**filter_input, hardlink=False)
    assert result["status"] == "COMPLETE"
    old = manifest(filter_input)
    assert_bundle(old)
    with pytest.raises(filtering.MissingRequiredMetricsError):
        filtering.filter_structures(**{**filter_input, "filter_expr": "absent > 0"})
    assert manifest(filter_input) == old
    assert_bundle(old)


@pytest.mark.parametrize("previous", [False, True])
def test_filter_partial_link_failure_does_not_publish_success(filter_input, monkeypatch, previous):
    if previous:
        filtering.filter_structures(**filter_input, hardlink=False)
        old = manifest(filter_input)
    real = os.link
    calls = []
    def fail_second(src, dst):
        calls.append(str(dst))
        if len(calls) == 2:
            raise OSError("injected second-link failure")
        return real(src, dst)
    monkeypatch.setattr(filtering.os, "link", fail_second)
    with pytest.raises(OSError, match="second-link"):
        filtering.filter_structures(**filter_input)
    if previous:
        assert manifest(filter_input) == old
        assert_bundle(old)
    else:
        assert manifest(filter_input)["status"] == "PARTIAL"
        assert not (filter_input["dst_dir"] / "filter_report.csv").exists()
        assert not list(filter_input["dst_dir"].rglob("*.cif"))


def test_filter_promotion_crash_fences_mixed_outputs(filter_input, monkeypatch):
    filtering.filter_structures(**filter_input, hardlink=False)
    real = os.replace
    def fail_report(src, dst):
        if Path(dst).name == "filter_report.csv":
            raise OSError("injected publication crash")
        return real(src, dst)
    monkeypatch.setattr(publication.os, "replace", fail_report)
    with pytest.raises(OSError, match="publication crash"):
        filtering.filter_structures(**filter_input, hardlink=False)
    assert manifest(filter_input)["status"] == "PARTIAL"


def test_filter_existing_wrong_content_is_not_success(filter_input):
    filtering.filter_structures(**filter_input, hardlink=False)
    old = manifest(filter_input)
    output = next(item for item in old["outputs"] if item["path"].endswith(".cif"))
    Path(output["path"]).write_text("unrelated existing file")
    with pytest.raises(FileExistsError):
        filtering.filter_structures(**filter_input)
    assert manifest(filter_input)["status"] == "PARTIAL"
    # A deliberately corrupted prior bundle is not a new successful publication.
    result = filtering.filter_structures(**filter_input, overwrite=True, hardlink=False)
    assert result["status"] == "COMPLETE"
    assert_bundle(manifest(filter_input))


def test_filter_existing_symlink_does_not_replace_source(filter_input):
    source = filter_input["db_path"].parent / "structure0.cif"
    destination = filter_input["dst_dir"] / "task/task__001_run__step_000__AB_model.cif"
    destination.parent.mkdir(parents=True)
    destination.symlink_to(source)
    inode = source.stat().st_ino
    filtering.filter_structures(**filter_input, hardlink=False)
    assert source.stat().st_ino == inode
    assert not destination.is_symlink()
    assert_bundle(manifest(filter_input))


def test_filter_missing_structures_and_exploratory_metrics_are_partial(filter_input):
    (filter_input["db_path"].parent / "structure0.cif").unlink()
    assert filtering.filter_structures(**filter_input)["status"] == "PARTIAL"
    assert manifest(filter_input)["status"] == "PARTIAL"
    result = filtering.filter_structures(**{**filter_input, "filter_expr": "absent > 0"},
                                         fail_on_missing_required=False)
    assert result["status"] == "PARTIAL"


def test_collect_csv_gate_and_archive_content_proof(source, tmp_path):
    run, arguments, _ = source
    raw = run / "raw_model_results" / "payload.txt"
    raw.parent.mkdir()
    raw.write_text("abc")
    csv = tmp_path / "export.csv"
    first = collect.collect_scores(**arguments, archive_dir=tmp_path / "archive", export_csv=csv)
    assert first["archived_created"] == 1
    assert json.loads(csv.with_name(csv.name + ".manifest.json").read_text())["status"] == "COMPLETE"
    old_csv = csv.read_bytes()
    with pytest.raises(collect.MissingRequiredMetricsError):
        collect.collect_scores(**arguments, required_metrics=["absent"], export_csv=csv)
    assert csv.read_bytes() == old_csv
    repeated = collect.collect_scores(**arguments, archive_dir=tmp_path / "archive")
    assert repeated["archived_skipped"] == 1
    archive = Path(repeated["archive_paths"][0])
    archive.write_bytes(b"corrupted archive")
    rebuilt = collect.collect_scores(**arguments, archive_dir=tmp_path / "archive")
    assert rebuilt["archived_created"] == 1
    stat = raw.stat()
    raw.write_text("def")
    os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert collect.collect_scores(**arguments, archive_dir=tmp_path / "archive")["archived_created"] == 1
