"""CPU contracts for the real public-interface final acceptance harness."""
import importlib.util
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf


PATH = Path(__file__).resolve().parents[1] / "scripts/validation/public_interface_final.py"


def _load():
    spec = importlib.util.spec_from_file_location("public_interface_final_test", PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def test_public_command_is_real_cli_budget_one():
    command = MODULE._public_command(
        input_path=Path("/case/input.pdb"),
        target_homology="/case/target.json:B",
        out_root=Path("/evidence/public"),
    )
    assert command[1:] == [
        "-m",
        "astrolabe.cli",
        "refine",
        "interface",
        "--input",
        "/case/input.pdb",
        "--target-homology",
        "/case/target.json:B",
        "--out-dir",
        "/evidence/public",
        "--budget",
        "1",
        "--target-chain",
        "B",
        "--binder-chain",
        "A",
    ]


def _prediction():
    scientific = {
        "sequence": "ACDE",
        "seed": 10079,
        "metrics": {"interface_ipae": 1.25},
        "holes": {"sum": -10.0},
        "geometry": {"target_ca_rmsd": 0.5},
        "score": 2.5,
    }
    return {
        "scores_normalized_sha256": "scores",
        "coordinates_sha256": "coordinates",
        "confidence_sha256": "confidence",
        "scientific": scientific,
    }


def test_strict_comparison_covers_arrays_coordinates_confidence_physics():
    left = _prediction()
    right = json.loads(json.dumps(left))
    assert MODULE._strict_comparison(left, right)["passed"]
    right["scientific"]["metrics"]["interface_ipae"] = 1.5
    comparison = MODULE._strict_comparison(left, right)
    assert not comparison["passed"]
    assert not comparison["checks"]["physics_and_result_scientific_exact"]


def test_owner_evidence_accepts_split_streams_and_rejects_polluted_stdout(tmp_path):
    service = tmp_path / "services/owned"
    service.mkdir(parents=True)
    stdout = service / "owner.jsonl"
    stderr = service / "owner.stderr.log"
    stdout.write_text('{"event":"service_ready"}\n')
    stderr.write_text("native diagnostic\n")
    (service / "lifecycle.json").write_text(json.dumps({
        "started_at": 1.0,
        "stdout_path": str(stdout),
        "stderr_path": str(stderr),
    }))
    evidence = MODULE._owner_evidence(tmp_path)
    assert evidence["states"][0]["stdout_path"] == str(stdout)
    assert evidence["states"][0]["stderr_path"] == str(stderr)

    stdout.write_text('{"event":"service_ready"}\nnative pollution\n')
    try:
        MODULE._owner_evidence(tmp_path)
    except MODULE.SHARED.AcceptanceError as error:
        assert "non-event data" in str(error)
    else:
        raise AssertionError("polluted stdout was accepted")


def test_source_manifest_pins_harness_and_shared_runtime():
    manifest = MODULE._source_manifest()
    assert "scripts/validation/public_interface_final.py" in manifest
    assert "scripts/validation/layout_then_public_gpu_acceptance.py" in manifest
    assert "astrolabe/evaluators/af3_lifecycle.py" in manifest


def test_baseline_config_does_not_prepopulate_beam_output(tmp_path, monkeypatch):
    public = tmp_path / "public"
    config = public / "prepared/case/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("placeholder: true\n")
    (public / "interface_plan.json").write_text(json.dumps({
        "cases": [{"config": "prepared/case/config.yaml"}],
    }))
    baseline = tmp_path / "baseline"
    service = tmp_path / "baseline-service"
    source = OmegaConf.create({
        "out_dir": str(public / "cases/case"),
        "protein_beam": {"af3_call_budget": {"ledger": str(public / "af3_calls.json")}},
    })

    from astrolabe import protein_beam

    monkeypatch.setattr(protein_beam, "load_config", lambda path: source.copy())
    monkeypatch.setattr(MODULE.SHARED, "_validate_service_beam_layout", lambda *args: None)
    prepared, evidence = MODULE._prepare_public_config(
        public_root=public,
        baseline_root=baseline,
        baseline_service_root=service,
    )
    assert prepared.out_dir == str(baseline)
    assert baseline.is_dir()
    assert list(baseline.iterdir()) == []
    assert Path(evidence["baseline_config"]) == tmp_path / "baseline-control/config.yaml"
