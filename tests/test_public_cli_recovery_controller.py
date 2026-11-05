from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import scripts.validation.public_cli_recovery_controller as controller


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_capsule_manifest_command_and_fingerprint_are_both_checked(tmp_path):
    manifest = {
        "schema": "astrolabe.gpu-acceptance.source-manifest/v1",
        "fingerprint": "expected-fingerprint",
        "files": {"run.py": "abc"},
    }
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "capsule_metadata.json").write_text(
        json.dumps({"status": "FROZEN_READY_NOT_LAUNCHED"}),
        encoding="utf-8",
    )
    evidence = controller.validate_capsule(
        tmp_path,
        expected_manifest_sha256=_sha256(path),
        expected_fingerprint="expected-fingerprint",
    )
    assert evidence["source_manifest_sha256"] == _sha256(path)
    assert evidence["source_fingerprint"] == "expected-fingerprint"

    original_sha256 = _sha256(path)
    manifest["fingerprint"] = "changed"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(controller.ControllerError, match="manifest changed"):
        controller.validate_capsule(
            tmp_path,
            expected_manifest_sha256=original_sha256,
            expected_fingerprint="expected-fingerprint",
        )
    with pytest.raises(controller.ControllerError, match="fingerprint changed"):
        controller.validate_capsule(
            tmp_path,
            expected_manifest_sha256=_sha256(path),
            expected_fingerprint="expected-fingerprint",
        )


def test_paper_overlay_differs_only_by_portable_output_storage(
    tmp_path,
    monkeypatch,
):
    original = tmp_path / "paper_inputs.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "seed": 17,
                "types": {"A": "p"},
                "seqs": {"A": "ACD"},
                "evaluator": {"term": {"num_seeds": 1}},
            }
        ),
        original,
    )
    overlay = tmp_path / "paper_inputs.portable.yaml"
    payload = OmegaConf.to_container(OmegaConf.load(original), resolve=False)
    payload["skip_complex_poses"] = True
    OmegaConf.save(OmegaConf.create(payload), overlay)
    monkeypatch.setattr(
        controller,
        "ORIGINAL_PAPER_INPUT_SHA256",
        _sha256(original),
    )

    evidence = controller.validate_paper_overlay(original, overlay)
    assert evidence["transport_only"] is True
    assert evidence["structural_difference"] == {
        "path": "$.skip_complex_poses",
        "value": True,
    }

    payload["steps"] = 3
    OmegaConf.save(OmegaConf.create(payload), overlay)
    with pytest.raises(controller.ControllerError, match="beyond skip_complex_poses"):
        controller.validate_paper_overlay(original, overlay)


def test_reusable_api_baseline_keeps_six_request_cli_plan():
    strategy = controller.select_baseline_strategy(
        SimpleNamespace(
            reuse=True,
            reasons=(),
            source_changes=(),
            config_differences=(),
            weights_match=True,
            runtime_image_match=True,
            rng_evidence_present=True,
            gpu_match=True,
        )
    )
    assert strategy.mode == "reuse"
    assert strategy.request_budget == 6


def test_gpu_only_mismatch_authorizes_two_request_default_refresh():
    strategy = controller.select_baseline_strategy(
        SimpleNamespace(
            reuse=False,
            reasons=("historical baseline GPU allocation differs",),
            source_changes=(),
            config_differences=(),
            weights_match=True,
            runtime_image_match=True,
            rng_evidence_present=True,
            gpu_match=False,
        )
    )
    assert strategy.mode == "fresh-default"
    assert strategy.request_budget == 8


@pytest.mark.parametrize(
    "overrides",
    [
        {"reasons": ("historical baseline GPU allocation differs", "extra")},
        {"source_changes": ({"path": "run.py"},)},
        {"config_differences": ({"path": "$.seed"},)},
        {"weights_match": False},
        {"runtime_image_match": False},
        {"rng_evidence_present": False},
        {"gpu_match": None},
    ],
)
def test_non_gpu_baseline_difference_refuses_model_execution(overrides):
    values = {
        "reuse": False,
        "reasons": ("baseline mismatch",),
        "source_changes": (),
        "config_differences": (),
        "weights_match": True,
        "runtime_image_match": True,
        "rng_evidence_present": True,
        "gpu_match": False,
    }
    values.update(overrides)
    with pytest.raises(controller.ControllerError, match="other than GPU"):
        controller.select_baseline_strategy(SimpleNamespace(**values))


def test_dry_run_prepares_plan_without_api_baseline_or_harness_import(
    tmp_path,
    monkeypatch,
):
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    manifest = {
        "schema": "astrolabe.gpu-acceptance.source-manifest/v1",
        "fingerprint": controller.EXPECTED_SOURCE_FINGERPRINT,
        "files": {},
    }
    manifest_path = capsule / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        controller,
        "EXPECTED_SOURCE_MANIFEST_SHA256",
        _sha256(manifest_path),
    )
    original = tmp_path / "paper_inputs.yaml"
    OmegaConf.save(OmegaConf.create({"seed": 17}), original)
    overlay = tmp_path / "paper_inputs.portable.yaml"
    OmegaConf.save(
        OmegaConf.create({"seed": 17, "skip_complex_poses": True}),
        overlay,
    )
    monkeypatch.setattr(
        controller,
        "ORIGINAL_PAPER_INPUT_SHA256",
        _sha256(original),
    )
    out_root = tmp_path / "out"
    status = controller.run_controller(
        capsule_root=capsule,
        expected_source_manifest_sha256=_sha256(manifest_path),
        expected_source_fingerprint=controller.EXPECTED_SOURCE_FINGERPRINT,
        out_root=out_root,
        api_baseline=tmp_path / "does-not-exist",
        failed_config=None,
        paper_inputs=original,
        paper_overlay=overlay,
        paper_baseline=tmp_path / "paper-baseline",
        profile=capsule / "profile.yaml",
        python_bin=Path("/unused/python"),
        af3_module_path=Path("/unused/af3"),
        timeout_s=1,
        dry_run=True,
    )
    assert status == 0
    payload = json.loads((out_root / "preflight.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["conditional_plan"]["api_gpu_match"]["request_budget"] == 6
    assert (
        payload["conditional_plan"]["api_gpu_mismatch_only"]["request_budget"]
        == 8
    )


def test_cpu_prelude_uses_apply_initialization_false_without_model_calls(
    tmp_path,
):
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    (capsule / "run.py").write_text(
        "from types import SimpleNamespace\n"
        "def prepare_pipeline_state(cfg, *, apply_initialization=True):\n"
        "    assert apply_initialization is False\n"
        "    return SimpleNamespace(\n"
        "        runtime_context=object(),\n"
        "        initializer_mode='test',\n"
        "        initializer_result=None,\n"
        "    )\n"
        "def _initialize_pipeline_math(cfg, prepared):\n"
        "    return object(), SimpleNamespace(current_oligo='A')\n",
        encoding="utf-8",
    )
    case_root = tmp_path / "failed-case"
    case_root.mkdir()
    config = case_root / "resume-input.yaml"
    OmegaConf.save(
        OmegaConf.create({"out_dir": str(tmp_path / "run"), "seed": 17}),
        config,
    )
    evidence = controller.run_cpu_prelude(
        capsule_root=capsule,
        config_path=config,
        out_root=tmp_path / "out",
        python_bin=Path(sys.executable),
        timeout_s=10,
    )
    assert evidence["status"] == "PASS"
    assert evidence["model_calls"] == 0
    assert evidence["result"]["apply_initialization"] is False
    assert evidence["result"]["initializer_result_is_none"] is True
    assert evidence["claims_before"] == evidence["claims_after"]
