"""CPU contracts for preserving the completed public root during continuation."""

import importlib.util
import json
from pathlib import Path
import sys


PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/validation/public_interface_continuation.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("public_interface_continuation_test", PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def test_resume_command_reuses_existing_public_cli_case(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    (public / "interface_plan.json").write_text(json.dumps({
        "input": "/case/D5056_D15.pdb",
        "homology": "/case/target.json",
        "cases": [{"output": "cases/D5056_D15", "target_chain": "B", "binder_chain": "A"}],
    }))
    command = MODULE._resume_command(public)
    assert MODULE._case_output(public) == public / "cases/D5056_D15"
    assert command[1:] == [
        "-m",
        "astrolabe.cli",
        "refine",
        "interface",
        "--input",
        "/case/D5056_D15.pdb",
        "--target-homology",
        "/case/target.json:B",
        "--out-dir",
        str(public),
        "--budget",
        "1",
        "--target-chain",
        "B",
        "--binder-chain",
        "A",
        "--resume",
    ]


def test_completed_resume_requires_unchanged_tree_events_and_zero_model():
    owner = {
        "events": {"counts": {"request_received": 1, "gpu_ok": 1}},
        "all_event_counts": {"model_load_start": 1},
    }
    passed = MODULE._resume_checks(
        returncode=0,
        tree_before={"result.json": "same"},
        tree_after={"result.json": "same"},
        owner_before=owner,
        owner_after=owner,
    )
    assert passed["passed"]
    assert passed["new_requests"] == 0
    assert passed["new_model_loads"] == 0

    failed = MODULE._resume_checks(
        returncode=0,
        tree_before={"result.json": "before"},
        tree_after={"result.json": "after"},
        owner_before=owner,
        owner_after={
            "events": {"counts": {"request_received": 2, "gpu_ok": 1}},
            "all_event_counts": {"model_load_start": 2},
        },
    )
    assert not failed["passed"]
    assert failed["new_requests"] == 1
    assert failed["new_model_loads"] == 1


def test_continuation_layout_keeps_baseline_config_outside_beam_state(tmp_path):
    paths = MODULE._continuation_paths(tmp_path / "continuation")
    assert paths["baseline"] == tmp_path / "continuation/baseline"
    assert paths["service"] == tmp_path / "continuation/baseline-service"
    assert paths["control"] == tmp_path / "continuation/baseline-control"
    assert paths["control"].parent == paths["baseline"].parent
    assert paths["control"] != paths["baseline"]


def test_source_manifest_pins_continuation_harness():
    manifest = MODULE._source_manifest()
    assert "scripts/validation/public_interface_continuation.py" in manifest
    assert "scripts/validation/public_interface_final.py" in manifest
    assert "astrolabe/interface.py" in manifest
