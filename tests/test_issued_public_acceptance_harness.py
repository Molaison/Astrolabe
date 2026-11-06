"""CPU contracts for issued recovery through the real public CLI."""
import importlib.util
import json
from pathlib import Path
import sys


PATH = Path(__file__).resolve().parents[1] / "scripts/validation/issued_public_acceptance.py"


def _load():
    spec = importlib.util.spec_from_file_location("issued_public_acceptance_test", PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def test_issued_command_is_real_public_cli_budget_one():
    command = MODULE._public_command(
        input_path=Path("/case/input.pdb"),
        target_homology="/case/target.json:B",
        out_root=Path("/evidence/public"),
        resume=True,
    )
    assert command[1:6] == [
        "-m",
        "astrolabe.cli",
        "refine",
        "interface",
        "--input",
    ]
    assert command[command.index("--budget") + 1] == "1"
    assert command[-1] == "--resume"


def test_issued_environment_arms_then_removes_receipt(tmp_path):
    instrumentation = tmp_path / "instrumentation"
    receipt = tmp_path / "receipt.json"
    armed = MODULE._environment(
        instrumentation,
        receipt=receipt,
        token="token-1",
    )
    assert armed["ASTROLABE_ACCEPTANCE_ISSUED_RECEIPT"] == str(receipt)
    assert armed["ASTROLABE_ACCEPTANCE_ISSUED_TOKEN"] == "token-1"
    assert armed["PYTHONPATH"].split(":")[0] == str(instrumentation)
    disarmed = MODULE._environment(
        instrumentation,
        receipt=None,
        token=None,
        )
    assert "ASTROLABE_ACCEPTANCE_ISSUED_RECEIPT" not in disarmed
    assert "ASTROLABE_ACCEPTANCE_ISSUED_TOKEN" not in disarmed


def test_snapshot_preserves_ledger_events_and_diagnostics(tmp_path):
    service_root = tmp_path / "public"
    service = service_root / "services/owned"
    case = service_root / "cases/D5056_D15"
    service.mkdir(parents=True)
    case.mkdir(parents=True)
    (service / "lifecycle.json").write_text(json.dumps({"started_at": 1.0}))
    (service / "owner.jsonl").write_text(
        '{"event":"request_received","request_id":1,"identity":{}}\n'
    )
    (service / "owner.stderr.log").write_text("diagnostic\n")
    (case / "af3_calls.json").write_text(json.dumps({"entries": {"root": {}}}))

    snapshot = MODULE._snapshot(service_root, case)

    assert snapshot["events"]["counts"]["request_received"] == 1
    assert snapshot["events"]["counts"]["gpu_start"] == 0
    assert snapshot["diagnostics"][str(service / "owner.stderr.log")] == b"diagnostic\n"
    assert json.loads(snapshot["ledger"])["entries"] == {"root": {}}


def test_instrumentation_claims_only_request_received():
    source = MODULE.ISSUED._instrumentation_source()
    assert 'event.get("event") != "request_received"' in source
    assert "signal.SIGSTOP" in source
    assert '"gpu_start"' not in source
