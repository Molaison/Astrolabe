from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "scripts" / "validation" / "issued_cut_gpu_acceptance.py"
CASE_FIXTURE = REPO_ROOT / "tests" / "data" / "protein_beam" / "02_fixture.yaml"


def _load_harness():
    name = "issued_cut_gpu_acceptance_harness_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_prepare_uses_one_call_and_sibling_service_layout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    monkeypatch.setattr(
        harness,
        "_coordinator_main",
        lambda arguments: pytest.fail("prepare-only started a model"),
    )
    out_root = tmp_path / "acceptance"
    assert harness.main(
        [
            "--out-root",
            str(out_root),
            "--case-config",
            str(CASE_FIXTURE),
            "--prepare-only",
        ]
    ) == 0
    plan = json.loads((out_root / "plan.json").read_text())
    assert plan["evaluation_budget_including_root"] == 1
    assert plan["af3_quota"] == {"max_total": 1, "max_per_case": 1}
    assert Path(plan["output"]).parent == Path(plan["service_root"]).parent
    assert Path(plan["output"]).name == "issued"
    assert Path(plan["service_root"]).name == "issued-service"
    assert plan["resume"] == {"retry_interrupted": False}
    assert "before executor.submit and gpu_start" in plan["cut"]
    assert "not interruption of executing GPU inference" in plan["cut"]
    assert plan["gpu_execution_interrupted"] is False
    assert not (out_root / "runs").exists()


def test_request_received_instrumentation_publishes_receipt_and_stops_owner(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    instrumentation = tmp_path / "instrumentation"
    instrumentation.mkdir()
    (instrumentation / "sitecustomize.py").write_text(
        harness._instrumentation_source(),
        encoding="utf-8",
    )
    receipt = tmp_path / "issued.receipt.json"
    stdout_path = tmp_path / "owner.jsonl"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(instrumentation)
    environment["ASTROLABE_ACCEPTANCE_ISSUED_RECEIPT"] = str(receipt)
    environment["ASTROLABE_ACCEPTANCE_ISSUED_TOKEN"] = "owned-token"
    code = (
        "import json,os\n"
        "os.write(1,(json.dumps({'event':'request_received',"
        "'request_id':1,'identity':{'candidate_id':0}})+'\\n').encode())\n"
    )
    with stdout_path.open("wb") as stdout:
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            env=environment,
            stdout=stdout,
            start_new_session=True,
        )
    try:
        deadline = time.monotonic() + 10
        while not receipt.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert receipt.is_file()
        cut = json.loads(receipt.read_text())
        assert cut["schema"] == "astrolabe.issued-gpu-cut/v1"
        assert cut["token"] == "owned-token"
        assert cut["service"]["pid"] == process.pid
        assert cut["request_event"]["request_id"] == 1
        identity = harness.SHARED._process_identity(process.pid)
        assert identity is not None and identity["state"] == "T"
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGCONT)
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


def test_issued_evidence_requires_one_running_root_and_no_gpu_terminal(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    prepared = harness._prepare(CASE_FIXTURE, tmp_path / "acceptance")
    prepared.output.mkdir(parents=True)
    (prepared.output / "af3_calls.json").write_text(
        json.dumps(
            {
                "entries": {
                    "scope:0:1": {
                        "candidate_id": 0,
                        "attempt": 1,
                        "status": "issued",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (prepared.output / "beam_state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "records": [{"id": 0, "status": "running"}],
            }
        ),
        encoding="utf-8",
    )
    service = prepared.service_root / "services" / "owned"
    service.mkdir(parents=True)
    process = subprocess.Popen(
        [sys.executable, "-c", "import signal; signal.pause()"],
        start_new_session=True,
    )
    try:
        identity = harness.SHARED._process_identity(process.pid)
        assert identity is not None
        (service / "lifecycle.json").write_text(
            json.dumps(
                {
                    "pid": process.pid,
                    "pid_start_ticks": identity["start_ticks"],
                    "boot_id": identity["boot_id"],
                    "status": "ready",
                }
            ),
            encoding="utf-8",
        )
        (service / "owner.jsonl").write_text(
            json.dumps(
                {
                    "event": "request_received",
                    "request_id": 1,
                    "pid": process.pid,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        evidence = harness._validate_issued_cut(
            prepared,
            {
                "schema": "astrolabe.issued-gpu-cut/v1",
                "token": "owned",
                "coordinator_pid": 123,
                "service": identity,
                "request_event": {"event": "request_received", "request_id": 1},
            },
            coordinator_pid=123,
            token="owned",
        )
        assert evidence["event_counts"]["request_received"] == 1
        assert evidence["event_counts"]["gpu_start"] == 0
        assert evidence["gpu_execution_interrupted"] is False
        assert "request_received was recorded" in evidence["proves"]
        assert "in-progress GPU inference" in evidence["does_not_prove"]
        assert evidence["ledger_entries"][0]["status"] == "issued"
        assert evidence["root_result_exists"] is False
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)


def test_resume_missing_terminal_stops_before_model_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    import astrolabe.evaluators.af3_lifecycle as lifecycle

    output = tmp_path / "runs" / "issued"
    service_root = tmp_path / "runs" / "issued-service"
    output.mkdir(parents=True)
    (output / "beam_state.json").write_text(
        json.dumps(
            {
                "status": "interrupted",
                "records": [{"id": 0, "status": "running"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        lifecycle,
        "recover_owned_services",
        lambda root: None,
    )
    monkeypatch.setattr(
        lifecycle,
        "OwnedAF3",
        lambda *args, **kwargs: pytest.fail(
            "issued-cut resume started a model owner"
        ),
    )
    arguments = type(
        "Arguments",
        (),
        {
            "config": CASE_FIXTURE,
            "output": output,
            "service_root": service_root,
            "resume": True,
            "source_fingerprint": harness._source_fingerprint(
                harness._source_manifest()
            ),
        },
    )()
    with pytest.raises(RuntimeError, match="lacks a terminal result"):
        harness._coordinator_main(arguments)


def test_resume_preserves_issued_cost_and_owner_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    prepared = harness._prepare(CASE_FIXTURE, tmp_path / "acceptance")
    ledger = prepared.output / "af3_calls.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "entries": {
                    "scope:0:1": {
                        "candidate_id": 0,
                        "attempt": 1,
                        "status": "issued",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    owner_log = (
        prepared.service_root / "services" / "owned" / "owner.jsonl"
    )
    owner_log.parent.mkdir(parents=True)
    owner_log.write_text(
        json.dumps({"event": "request_received", "request_id": 1}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(harness.SHARED, "_full_run_reasons", lambda *args: [])
    monkeypatch.setattr(
        harness,
        "_run_issued_cut",
        lambda *args, **kwargs: ({}, {"event_counts": {"request_received": 1}}),
    )
    monkeypatch.setattr(
        harness,
        "_run_resume",
        lambda *args, **kwargs: {"returncode": 1, "expected_error": "unknown"},
    )
    summary = harness._execute(prepared, timeout_s=1)
    assert summary["status"] == "PASS"
    assert summary["issues"] == []
    assert json.loads(ledger.read_text())["entries"]["scope:0:1"]["status"] == "issued"
