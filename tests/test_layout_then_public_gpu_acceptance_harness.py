from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    REPO_ROOT
    / "scripts"
    / "validation"
    / "layout_then_public_gpu_acceptance.py"
)


def _load_harness():
    name = "layout_then_public_gpu_acceptance_harness_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_public_run(output: Path) -> None:
    output.mkdir(parents=True)
    (output / "interface_plan.json").write_text("{}", encoding="utf-8")
    (output / "interface_results.json").write_text(
        json.dumps([{"status": "complete", "budget_complete": True}]),
        encoding="utf-8",
    )
    service = output / "services" / "owned"
    service.mkdir(parents=True)
    (service / "lifecycle.json").write_text(
        json.dumps({"started_at": 1.0, "status": "drained"}),
        encoding="utf-8",
    )
    events = [
        {"event": "request_received", "request_id": 1},
        {"event": "gpu_start", "request_id": 1},
        {"event": "gpu_end", "request_id": 1, "ok": True},
    ]
    (service / "owner.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    case = output / "cases" / "D5056_D15"
    ledger = case / "af3_calls.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps({"entries": {"scope:0:1": {"status": "issued"}}}),
        encoding="utf-8",
    )
    candidate = case / "candidates" / "000000"
    candidate.mkdir(parents=True)
    (candidate / "result.json").write_text("{}", encoding="utf-8")
    (candidate / "scores.pkl").write_bytes(b"snapshot")


def test_commands_use_real_layout_and_frontend_paths(tmp_path: Path) -> None:
    harness = _load_harness()
    layout = harness._layout_command(
        out_root=tmp_path / "layout",
        case_config=tmp_path / "case.yaml",
        prepare_only=True,
    )
    assert layout[1].endswith("isolated_gpu_acceptance.py")
    assert "--prepare-only" in layout
    public = harness._public_command(
        input_path=tmp_path / "input.pdb",
        target_homology=str(tmp_path / "target.json") + ":B",
        out_root=tmp_path / "public",
        resume=True,
    )
    assert public[1:4] == ["-m", "astrolabe.cli", "refine"]
    assert "interface" in public
    assert public[public.index("--budget") + 1] == "1"
    assert "--resume" in public


def test_public_first_run_requires_one_complete_request(tmp_path: Path) -> None:
    harness = _load_harness()
    output = tmp_path / "public"
    _write_public_run(output)
    harness._validate_public_first(output)


def test_layout_failure_blocks_public_phase(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    labels = []

    def fail_layout(command, *, label, log_root, timeout_s):
        labels.append(label)
        return 1

    monkeypatch.setattr(harness, "_run", fail_layout)
    with pytest.raises(harness.SHARED.AcceptanceError, match="layout"):
        harness._execute(
            case_config=tmp_path / "case.yaml",
            input_path=tmp_path / "input.pdb",
            target_homology="/case/target.json:B",
            out_root=tmp_path / "acceptance",
            timeout_s=1,
        )
    assert labels == ["layout"]


def test_completed_resume_adds_zero_native_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    labels = []

    def run(command, *, label, log_root, timeout_s):
        labels.append(label)
        if label == "layout":
            output = Path(command[command.index("--out-root") + 1])
            output.mkdir(parents=True)
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "comparison": {"native_call_count": 4},
                    }
                ),
                encoding="utf-8",
            )
        elif label == "public-root":
            output = Path(command[command.index("--out-dir") + 1])
            _write_public_run(output)
        return 0

    monkeypatch.setattr(harness, "_run", run)
    summary = harness._execute(
        case_config=tmp_path / "case.yaml",
        input_path=tmp_path / "input.pdb",
        target_homology="/case/target.json:B",
        out_root=tmp_path / "acceptance",
        timeout_s=1,
    )
    assert labels == ["layout", "public-root", "public-resume"]
    assert summary["status"] == "PASS"
    assert summary["layout_native_calls"] == 4
    assert summary["public_root_native_calls"] == 1
    assert summary["public_completed_resume_native_calls"] == 0
