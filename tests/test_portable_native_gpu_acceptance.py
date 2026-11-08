from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    REPO_ROOT
    / "scripts"
    / "validation"
    / "portable_native_gpu_acceptance.py"
)
PROFILE = REPO_ROOT / "config" / "container_af3_smoke.yaml"


def _load_harness():
    module_name = "portable_native_gpu_acceptance_test_module"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_matrix_uses_public_api_and_exact_six_request_budget() -> None:
    harness = _load_harness()
    matrix = harness.acceptance_matrix((harness.RAW_PREDICTION_CUT,))

    assert [case.name for case in matrix] == [
        "native-in-process",
        "public-api-durable",
        "public-api-after-raw-prediction",
    ]
    assert harness.request_budget((harness.RAW_PREDICTION_CUT,)) == 6
    assert matrix[0].execution == "in_process"
    assert matrix[1].execution == "daemon"
    assert matrix[1].recovery_resume is False
    assert matrix[2].cut == harness.RAW_PREDICTION_CUT
    assert "astrolabe.app.Astrolabe.run" in matrix[1].frontdoor
    assert "run._run_pipeline" in matrix[1].frontdoor
    with pytest.raises(ValueError, match="unsupported"):
        harness.acceptance_matrix(("after-decision",))


def test_explicit_skip_normal_public_mode_has_exact_four_request_budget() -> None:
    harness = _load_harness()
    matrix = harness.acceptance_matrix(
        (harness.RAW_PREDICTION_CUT,),
        include_normal_public=False,
    )

    assert [case.name for case in matrix] == [
        "native-in-process",
        "public-api-after-raw-prediction",
    ]
    assert harness.request_budget(
        (harness.RAW_PREDICTION_CUT,),
        include_normal_public=False,
    ) == 4
    assert harness.request_budget(
        (harness.RAW_PREDICTION_CUT,),
        include_normal_public=True,
    ) == 6


def test_case_configs_keep_frozen_fixture_and_explicit_durable_recovery(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    case = harness.acceptance_matrix((harness.RAW_PREDICTION_CUT,))[1]
    cfg = harness._case_config(
        profile=PROFILE,
        case=case,
        out_dir=tmp_path / "output",
    )

    assert cfg.seed == 17
    assert cfg.steps == 2
    assert cfg.repeats == 1
    assert cfg.evaluator.alphafold3.num_samples == 1
    assert cfg.evaluator.alphafold3.num_seeds == 1
    assert cfg.evaluator.alphafold3.num_recycles == 1
    assert cfg.evaluator.alphafold3.run_data_pipeline is False
    assert cfg.evaluator.alphafold3.execution == "daemon"
    assert OmegaConf.to_container(cfg.recovery, resolve=True) == {
        "mode": "durable",
        "resume": False,
    }


def test_resume_config_uses_persisted_run_root_without_rewriting_it(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    case = harness.acceptance_matrix((harness.RAW_PREDICTION_CUT,))[1]
    run_root = tmp_path / "parsed" / "task" / "001_run"
    run_root.mkdir(parents=True)
    cfg = harness._case_config(
        profile=PROFILE,
        case=case,
        out_dir=run_root,
    )
    persisted = run_root / "config.yaml"
    harness._write_config(cfg, persisted)
    before = persisted.read_bytes()
    resume_path = tmp_path / "resume.yaml"

    resumed = harness._prepare_resume_config(run_root, resume_path)

    assert persisted.read_bytes() == before
    assert resume_path.is_file()
    assert Path(resumed.out_dir).resolve() == run_root.resolve()
    assert resumed.recovery.mode == "durable"
    assert resumed.recovery.resume is True
    assert harness._run_persisted_run(tmp_path / "parsed") == run_root.resolve()


def test_persisted_root_mismatch_is_rejected(tmp_path: Path) -> None:
    harness = _load_harness()
    case = harness.acceptance_matrix((harness.RAW_PREDICTION_CUT,))[1]
    actual = tmp_path / "actual"
    actual.mkdir()
    cfg = harness._case_config(
        profile=PROFILE,
        case=case,
        out_dir=actual,
    )
    cfg.out_dir = str(tmp_path / "forged")
    harness._write_config(cfg, actual / "config.yaml")

    with pytest.raises(harness.AcceptanceError, match="does not identify"):
        harness._run_persisted_run(actual)


def test_raw_cut_accounting_requires_one_initial_and_one_remaining_call() -> None:
    harness = _load_harness()
    cut = [{"step": 0, "status": "ok"}]
    resume = [{"step": 1, "status": "ok"}]

    evidence = harness.validate_raw_cut_accounting(cut, resume)

    assert evidence == {
        "cut_model_requests": 1,
        "resume_model_requests": 1,
        "remaining_model_requests": 1,
        "duplicate_model_requests": 0,
        "cut_steps": [0],
        "resume_steps": [1],
        "total_case_model_requests": 2,
    }
    with pytest.raises(harness.AcceptanceError, match="duplicate|unexpected"):
        harness.validate_raw_cut_accounting([{"step": 0}], [{"step": 0}])
    with pytest.raises(harness.AcceptanceError, match="exactly one remaining"):
        harness.validate_raw_cut_accounting(
            [{"step": 0}],
            [{"step": 1}, {"step": 1}],
        )


def test_runtime_cut_fires_only_after_named_call_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    from astrolabe.design.runtime import Runtime

    calls = []

    def fake_call(self, named_round, operation, request, *, work_id=None):
        calls.append((named_round, operation, request, work_id))
        return {"named_round": named_round}

    class ForcedExit(BaseException):
        pass

    monkeypatch.setattr(Runtime, "call", fake_call)
    monkeypatch.setattr(
        harness.os,
        "_exit",
        lambda code: (_ for _ in ()).throw(ForcedExit(code)),
    )
    marker = tmp_path / "cut-marker.json"
    harness._install_runtime_cut("root/raw_predict", marker)

    assert Runtime.call(  # type: ignore[misc]
        object(),
        "root/finalize_state",
        "operation",
        "request",
    ) == {"named_round": "root/finalize_state"}
    with pytest.raises(ForcedExit) as captured:
        Runtime.call(  # type: ignore[misc]
            object(),
            "root/raw_predict",
            "operation",
            "request",
        )

    assert captured.value.args == (harness.CUT_EXIT,)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["boundary"] == "root/raw_predict"
    assert calls == [
        ("root/finalize_state", "operation", "request", None),
        ("root/raw_predict", "operation", "request", None),
    ]


def test_public_worker_calls_the_application_entrypoint_not_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    import astrolabe.app as app_module

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "task_name: portable-worker-test\n",
        encoding="utf-8",
    )
    observed = []

    class PublicApp:
        def __init__(self, config):
            observed.append(("construct", config.task_name))

        def run(self):
            observed.append(("run",))

    monkeypatch.setattr(app_module, "Astrolabe", PublicApp)

    assert harness._worker_main(
        mode="public-api",
        config_path=config_path,
    ) == 0
    assert observed == [
        ("construct", "portable-worker-test"),
        ("run",),
    ]


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    harness = _load_harness()
    out_root = tmp_path / "acceptance"
    out_root.mkdir()
    protected = out_root / "old-failed-result.json"
    protected.write_text('{"status":"failed"}\n', encoding="utf-8")
    before = protected.read_bytes()

    with pytest.raises(harness.AcceptanceError, match="refusing to overwrite"):
        harness._reject_existing_output(out_root)

    assert protected.read_bytes() == before
    assert sorted(path.name for path in out_root.iterdir()) == [
        "old-failed-result.json"
    ]


def test_dry_run_records_six_request_budget_without_gpu_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    unified = harness._load_unified()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run must not inspect a GPU or model weights")

    monkeypatch.setattr(unified, "_environment_manifest", forbidden)
    monkeypatch.setattr(unified, "_nvidia_manifest", forbidden)
    out_root = tmp_path / "dry"

    assert harness.main(
        [
            "--out-root",
            str(out_root),
            "--profile",
            str(PROFILE),
            "--dry-run",
        ]
    ) == 0
    plan = json.loads((out_root / "plan.json").read_text(encoding="utf-8"))
    assert plan["status"] == "dry-run"
    assert plan["request_budget"] == 6
    assert plan["hard_request_cap"] == 8
    assert plan["no_retry"] is True
    assert [item["name"] for item in plan["cases"]] == [
        "native-in-process",
        "public-api-durable",
        "public-api-after-raw-prediction",
    ]


def test_dry_run_skip_normal_public_records_exact_four_request_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    unified = harness._load_unified()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run must not inspect a GPU or model weights")

    monkeypatch.setattr(unified, "_environment_manifest", forbidden)
    monkeypatch.setattr(unified, "_nvidia_manifest", forbidden)
    out_root = tmp_path / "dry-four"

    assert harness.main(
        [
            "--out-root",
            str(out_root),
            "--profile",
            str(PROFILE),
            "--dry-run",
            "--skip-normal-public",
        ]
    ) == 0
    plan = json.loads((out_root / "plan.json").read_text(encoding="utf-8"))
    assert plan["status"] == "dry-run"
    assert plan["request_budget"] == 4
    assert plan["include_normal_public"] is False
    assert [item["name"] for item in plan["cases"]] == [
        "native-in-process",
        "public-api-after-raw-prediction",
    ]


def test_public_astrolabe_cold_route_builds_recipe_before_any_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe import app as app_module
    from astrolabe.configuration import load_composed_config
    from astrolabe.core import global_config
    from astrolabe.design.portable_native_builder import PortableNativeRecipe
    from omegaconf import open_dict

    global_config.reset()
    try:
        model_dir = tmp_path / "af3-model"
        model_dir.mkdir()
        (model_dir / "weights.bin").write_bytes(b"portable-cold-route-test")

        cfg = load_composed_config("config/container_af3_smoke.yaml")
        with open_dict(cfg):
            cfg.task_name = "portable-public-cold-route"
            cfg.out_dir = str(tmp_path / "run")
            cfg.raw_model_results_dir = str(tmp_path / "raw")
            cfg.steps = 2
            cfg.repeats = 1
            cfg.evaluator.evaluator_terms.Alphafold3.model_dir = str(model_dir)
            cfg.recovery = {"mode": "durable", "resume": False}

        app_module._load_run_module.cache_clear()
        run_module = app_module._load_run_module()
        assert sys.modules["astrolabe_run_module"] is run_module

        forbidden = []

        def forbid(name):
            def fail(*_args, **_kwargs):
                forbidden.append(name)
                raise AssertionError(f"cold public route reached {name}")

            return fail

        monkeypatch.setattr(run_module, "build_mutators", forbid("build_mutators"))
        monkeypatch.setattr(
            run_module,
            "construct_evaluators",
            forbid("construct_evaluators"),
        )
        monkeypatch.setattr(
            run_module,
            "initialize_predockers",
            forbid("initialize_predockers"),
        )
        monkeypatch.setattr(run_module, "setup_logging", lambda *_args, **_kwargs: None)

        built = []

        class StopBeforeModelExecution(RuntimeError):
            pass

        def stop_before_run(recipe):
            built.append(recipe)
            raise StopBeforeModelExecution("cold route stopped before recipe.run")

        monkeypatch.setattr(PortableNativeRecipe, "run", stop_before_run)

        with pytest.raises(
            StopBeforeModelExecution,
            match="stopped before recipe.run",
        ):
            app_module.Astrolabe(cfg).run()

        assert forbidden == []
        assert len(built) == 1
        recipe = built[0]
        assert recipe.flow.evaluator.owner.ready is False
        assert recipe.flow.mutator.owner.ready is False
        assert any(
            Path(item["path"]).name == "weights.bin"
            for item in recipe.asset_manifest["files"]
        )
    finally:
        app_module._load_run_module.cache_clear()
        global_config.reset()


def test_fresh_capture_sitecustomize_supports_entry_claims_and_digests(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    unified = harness._load_unified()
    case_root = tmp_path / "capture-case"
    capture, claims, env = unified._install_capture_environment(case_root)
    script = """
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import jsonschema

capsule = Path(os.environ["ASTROLABE_CAPSULE_ROOT"])
entry = capsule / "scripts" / "validation" / "portable_native_gpu_acceptance.py"
spec = importlib.util.spec_from_file_location("portable_entry_probe", entry)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
unified = module._load_unified()
manifest_path = Path(os.environ["ASTROLABE_SOURCE_MANIFEST"])
manifest = unified._load_source_manifest(manifest_path)
current = unified._read_source_manifest()
current_files = dict(current.files)
assert all(current_files.get(path) == digest for path, digest in manifest.files)
claims = Path(os.environ["ASTROLABE_ACCEPTANCE_CLAIMS"])
assert claims.is_dir()
capture = Path(os.environ["ASTROLABE_ACCEPTANCE_CAPTURE"])
assert unified._read_capture_records(capture) == []
assert len(hashlib.sha256(manifest_path.read_bytes()).hexdigest()) == 64
print(json.dumps({
    "jsonschema": jsonschema.__version__,
    "files": len(manifest.files),
    "fingerprint": manifest.fingerprint,
}, sort_keys=True))
"""
    probe_env = dict(env)
    probe_env["ASTROLABE_CAPSULE_ROOT"] = str(unified.REPO_ROOT)
    manifest_path = tmp_path / "source_manifest.json"
    unified._atomic_json(
        manifest_path,
        unified._read_source_manifest().as_dict(),
    )
    probe_env["ASTROLABE_SOURCE_MANIFEST"] = str(manifest_path)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=unified.REPO_ROOT,
        env=probe_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["files"] > 0
    assert len(payload["fingerprint"]) == 64
    assert capture.parent == case_root
    assert claims == case_root / "call-claims"
