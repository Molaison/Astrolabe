from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = REPO_ROOT / "scripts" / "validation" / "current_paired_acceptance.py"


def _load_driver():
    name = "current_paired_acceptance_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_profile_fixture(tmp_path: Path) -> tuple[Path, Path]:
    profile_root = tmp_path / "profile"
    profile_root.mkdir()
    first_root = profile_root / "first-root"
    structure = first_root / "model.cif"
    snapshot = first_root / "scores.pkl"
    confidence = first_root / "confidence.json"
    owner = first_root / "owner.jsonl"
    ledger = first_root / "af3_calls.json"
    environment = first_root / "environment.json"
    source_manifest = first_root / "source_manifest.json"
    result_path = first_root / "result.json"
    weights = profile_root / "af3.bin"
    target_structure = profile_root / "target.pdb"
    target_homology = profile_root / "homology.json"
    template_config = profile_root / "template.yaml"
    profile_case = profile_root / "case.yaml"
    historical = profile_root / "historical-result.json"

    first_root.mkdir()
    structure.write_text("data_model\n", encoding="utf-8")
    snapshot.write_bytes(b"scores")
    confidence.write_text('{"pae": [[0.0]]}\n', encoding="utf-8")
    owner.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"event": "request_received"},
                {"event": "gpu_start"},
                {"event": "gpu_end", "ok": True, "request_id": 1},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    ledger.write_text(
        json.dumps({"entries": {"scope:0:1": {"status": "issued"}}}),
        encoding="utf-8",
    )
    environment.write_text("{}\n", encoding="utf-8")
    source_manifest.write_text("{}\n", encoding="utf-8")
    weights.write_bytes(b"weights")
    target_structure.write_bytes(b"target")
    target_homology.write_bytes(b"homology")
    historical.write_text("{}\n", encoding="utf-8")
    old_reference = str(historical)
    new_reference = str(result_path)
    template_config.write_text(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "seed": 10079,
                    "seqs": {"A": "ACDE", "B": "FGHI"},
                    "protein_beam": {"root_parity_reference": old_reference},
                }
            )
        ),
        encoding="utf-8",
    )
    profile_case.write_text(
        template_config.read_text(encoding="utf-8").replace(
            old_reference, new_reference
        ),
        encoding="utf-8",
    )
    result_path.write_text(
        json.dumps(
            {
                "id": 0,
                "parent_id": None,
                "status": "done",
                "evaluation_origin": "fresh_af3",
                "sequence": "ACDE",
                "seed": 10079,
                "af3_call_token": "scope:0:1",
                "metrics": {"interface_ipae": 1.0},
                "structure_path": str(structure),
                "snapshot": str(snapshot),
            }
        ),
        encoding="utf-8",
    )

    paths = {
        "root_result": result_path,
        "root_structure": structure,
        "root_snapshot": snapshot,
        "root_confidence": confidence,
        "root_owner_log": owner,
        "root_call_ledger": ledger,
        "root_environment": environment,
        "root_source_manifest": source_manifest,
        "weights": weights,
        "target_structure": target_structure,
        "target_homology": target_homology,
        "scientific_template_config": template_config,
        "profile_case_config": profile_case,
        "validation_driver": DRIVER_PATH,
    }
    profile = {
        "schema": "astrolabe.current-runtime-paired-profile/v1",
        "profile": "current_runtime_paired_v1",
        "assets": {
            key: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for key, path in paths.items()
        },
        "fresh_pair": {
            "first_half": {
                "root_result": str(result_path),
                "confidence": str(confidence),
                "owner_log": str(owner),
                "call_ledger": str(ledger),
                "environment": str(environment),
                "source_manifest": str(source_manifest),
            },
            "expected_owner_events": {
                "request_received": 1,
                "gpu_start": 1,
                "gpu_end": 1,
            },
        },
        "runtime": {"weights_path": str(weights)},
        "inputs": {
            "target_structure": str(target_structure),
            "target_homology": str(target_homology),
        },
        "scientific_template": {
            "config": str(template_config),
            "profile_case": str(profile_case),
        },
        "validation_driver": {"path": str(DRIVER_PATH.resolve())},
        "historical_reference": {"root_result": str(historical)},
    }
    manifest = profile_root / "profile.json"
    manifest.write_text(json.dumps(profile), encoding="utf-8")
    return manifest, profile_case


def test_profile_accepts_only_root_reference_change(tmp_path: Path) -> None:
    driver = _load_driver()
    manifest, _ = _write_profile_fixture(tmp_path)
    profile, evidence = driver._validate_static(
        manifest,
        out_root=tmp_path / "execution",
    )
    assert profile["profile"] == "current_runtime_paired_v1"
    assert evidence["self_comparison"] is False
    assert evidence["scientific_config_differences"] == [
        {
            "path": "protein_beam.root_parity_reference",
            "template": profile["historical_reference"]["root_result"],
            "profile": profile["fresh_pair"]["first_half"]["root_result"],
        }
    ]
    assert evidence["historical_reference"]["strict_parity_passed"] is False


def test_profile_rejects_additional_scientific_change(tmp_path: Path) -> None:
    driver = _load_driver()
    manifest, profile_case = _write_profile_fixture(tmp_path)
    profile = json.loads(manifest.read_text())
    profile_case.write_text(
        profile_case.read_text() + "\nnew_scientific_value: 1\n",
        encoding="utf-8",
    )
    asset = profile["assets"]["profile_case_config"]
    asset["sha256"] = _sha256(profile_case)
    manifest.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(driver.ProfileError, match="root_parity_reference"):
        driver._validate_static(manifest, out_root=tmp_path / "execution")


def test_profile_rejects_reference_inside_execution_root(tmp_path: Path) -> None:
    driver = _load_driver()
    manifest, _ = _write_profile_fixture(tmp_path)
    profile = json.loads(manifest.read_text())
    execution = Path(profile["fresh_pair"]["first_half"]["root_result"]).parent.parent
    with pytest.raises(driver.ProfileError, match="cannot reside under"):
        driver._validate_static(manifest, out_root=execution)
