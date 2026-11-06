from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf, open_dict
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    REPO_ROOT
    / "scripts"
    / "validation"
    / "public_cli_recovery_gpu_acceptance.py"
)
PROFILE = REPO_ROOT / "config" / "container_af3_smoke.yaml"
PAPER_INPUT = (
    REPO_ROOT
    / "outputs"
    / "unified-gpu-acceptance"
    / "final_callers_v2"
    / "attempt_001"
    / "cases"
    / "paper-cli"
    / "paper_inputs.yaml"
)


def _load_harness():
    module_name = "public_cli_recovery_gpu_acceptance_test_module"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(module_name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _command_overrides(command: list[str]) -> list[str]:
    index = command.index("--overrides")
    return shlex.split(command[index + 1])


def test_public_commands_use_real_cli_and_transport_only_overrides(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    python_bin = Path("/venv/bin/python")
    default_config = tmp_path / "default.yaml"
    paper_input = tmp_path / "paper inputs.yaml"
    root = tmp_path / "parsed run root"
    task_name = harness.TASK_NAME

    default = harness.public_cli_command(
        case=harness._case_spec("default-cli"),
        python_bin=python_bin,
        config_path=default_config,
        paper_input=None,
        task_name=task_name,
        out_dir=root,
        resume=True,
    )
    paper = harness.public_cli_command(
        case=harness._case_spec("paper-cli"),
        python_bin=python_bin,
        config_path=None,
        paper_input=paper_input,
        task_name=task_name,
        out_dir=root,
        resume=True,
    )

    assert default[:4] == [
        str(python_bin),
        "-m",
        "astrolabe.cli",
        "run",
    ]
    assert default[default.index("--standalone") + 1] == str(default_config)
    assert "--api-worker" not in default
    assert "--internal-worker" not in default

    assert paper[:6] == [
        str(python_bin),
        "-m",
        "astrolabe.cli",
        "paper",
        "run",
        "evaluate-af3",
    ]
    assert paper[paper.index("--inputs") + 1] == str(paper_input)
    assert "--api-worker" not in paper
    assert "--internal-worker" not in paper
    assert paper[paper.index("--task-name") + 1] == task_name

    expected = [
        "recovery.mode=durable",
        "recovery.resume=true",
        f"out_dir={root}",
        "skip_complex_poses=true",
    ]
    assert _command_overrides(default) == expected
    assert _command_overrides(paper) == expected


def test_initial_and_resume_commands_differ_only_in_transport_overrides(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    case = harness._case_spec("paper-cli")
    paper_input = tmp_path / "paper_inputs.yaml"
    initial_out = tmp_path / "output"
    persisted_root = initial_out / "task" / "001_run"
    initial = harness.public_cli_command(
        case=case,
        python_bin=Path("/venv/bin/python"),
        config_path=None,
        paper_input=paper_input,
        task_name="paper-task",
        out_dir=initial_out,
        resume=False,
    )
    resume = harness.public_cli_command(
        case=case,
        python_bin=Path("/venv/bin/python"),
        config_path=None,
        paper_input=paper_input,
        task_name="paper-task",
        out_dir=persisted_root,
        resume=True,
        cli_out_dir=initial_out,
    )

    report = harness.validate_command_transport_diff(
        initial_command=initial,
        resume_command=resume,
    )

    assert report["status"] == "PASS"
    assert report["changed_overrides"] == ["out_dir", "recovery.resume"]
    assert report["all_non_override_argv_equal"] is True


def test_paper_and_default_dry_run_accept_their_real_cli_parsers(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    paper_source = tmp_path / "paper_inputs.yaml"
    paper_source.write_bytes(PAPER_INPUT.read_bytes())
    paper_before = paper_source.read_bytes()
    paper_root = tmp_path / "paper run root"
    paper_task = f"{harness.TASK_NAME}-paper-cli"
    paper_command = harness.public_cli_command(
        case=harness._case_spec("paper-cli"),
        python_bin=Path(sys.executable),
        config_path=None,
        paper_input=paper_source,
        task_name=paper_task,
        out_dir=paper_root,
        resume=True,
        dry_run=True,
    )
    paper_process = subprocess.run(
        paper_command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert paper_process.returncode == 0, paper_process.stderr
    paper_cfg = OmegaConf.create(paper_process.stdout)
    assert paper_cfg.task_name == paper_task
    assert Path(paper_cfg.out_dir).resolve() == paper_root.resolve()
    assert paper_cfg.recovery.mode == "durable"
    assert paper_cfg.recovery.resume is True
    assert paper_cfg.seqs.A == "+ACDEFGHIKL@shared"
    assert paper_cfg.seqs.B == "shared +MNPQRSTVWY"
    assert paper_source.read_bytes() == paper_before

    default_root = tmp_path / "default run root"
    default_cfg = harness._case_config(
        case=harness._case_spec("default-cli"),
        profile=PROFILE,
        out_dir=default_root,
        task_name=f"{harness.TASK_NAME}-default-cli",
    )
    default_input = tmp_path / "default-input.yaml"
    harness._load_unified()._save_standalone(default_cfg, default_input)
    default_command = harness.public_cli_command(
        case=harness._case_spec("default-cli"),
        python_bin=Path(sys.executable),
        config_path=default_input,
        paper_input=None,
        task_name=f"{harness.TASK_NAME}-default-cli",
        out_dir=default_root,
        resume=True,
        dry_run=True,
    )
    default_process = subprocess.run(
        default_command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert default_process.returncode == 0, default_process.stderr
    composed = OmegaConf.create(default_process.stdout)
    assert composed.task_name == f"{harness.TASK_NAME}-default-cli"
    assert Path(composed.out_dir).resolve() == default_root.resolve()
    assert composed.recovery.mode == "durable"
    assert composed.recovery.resume is True
    assert composed.seqs.A == "ACDEFGHIKLMNPQRSTVWY"


def test_resume_diff_allows_only_transport_and_recovery_changes(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    persisted = OmegaConf.create(
        {
            "task_name": "semantic-task",
            "out_dir": str(tmp_path / "parent"),
            "repeat_index": 0,
            "raw_model_results_dir": str(tmp_path / "raw"),
            "seed": 17,
            "seqs": {"A": "ACDE"},
            "evaluator": {"alphafold3": {"num_samples": 1}},
            "recovery": {"mode": "durable", "resume": False},
        }
    )
    resumed_root = tmp_path / "parent" / "001_run"
    resumed = OmegaConf.create(OmegaConf.to_container(persisted, resolve=False))
    with open_dict(resumed):
        resumed.out_dir = str(resumed_root)
        resumed.recovery.resume = True
        if "repeat_index" in resumed:
            del resumed.repeat_index
        if "raw_model_results_dir" in resumed:
            del resumed.raw_model_results_dir

    report = harness.validate_resume_override_diff(
        persisted_cfg=persisted,
        resume_cfg=resumed,
        persisted_root=resumed_root,
    )

    assert report["status"] == "PENDING_CANONICAL_RUNTIME_CHECK"
    assert report["task_name"] == "semantic-task"
    assert report["recovery"] == {"mode": "durable", "resume": True}

    changed = OmegaConf.create(OmegaConf.to_container(resumed, resolve=False))
    changed.task_name = "different-task"
    with pytest.raises(harness.AcceptanceError, match="task_name"):
        harness.validate_resume_override_diff(
            persisted_cfg=persisted,
            resume_cfg=changed,
            persisted_root=resumed_root,
        )


def test_budget_is_reserved_before_launch_and_cannot_exceed_hard_cap(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    assert harness.request_budget(
        fresh_default=False,
        fresh_paper=False,
    ) == 4
    assert harness.request_budget(
        fresh_default=False,
        fresh_paper=True,
    ) == 6
    with pytest.raises(
        harness.AcceptanceError,
        match="refusing to recreate it",
    ):
        harness.request_budget(
            fresh_default=True,
            fresh_paper=True,
        )

    ledger = harness.RequestBudget(tmp_path / "budget.json")
    ledger.reserve("cut:default", 1)
    ledger.reserve("resume:default", 1)
    ledger.reserve("fresh-baseline:native-pair", 2)
    ledger.reserve("cut:paper", 1)
    ledger.reserve("resume:paper", 1)
    with pytest.raises(harness.AcceptanceError, match="budget exceeded"):
        ledger.reserve("forbidden-extra", 3)

    payload = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert payload["limit"] == 8
    assert payload["reserved_before_launch"] == 6
    assert payload["remaining"] == 2
    assert payload["no_retry"] is True


def test_public_case_selection_defaults_to_both_and_accepts_paper_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    parser = harness._build_argument_parser()

    assert harness._selected_cases(
        parser.parse_args(["--out-root", str(tmp_path / "all")]).cases
    ) == harness.SUPPORTED_CASES
    paper_only = parser.parse_args(
        [
            "--out-root",
            str(tmp_path / "paper"),
            "--cases",
            "paper-cli",
        ]
    )
    assert harness._selected_cases(paper_only.cases) == ("paper-cli",)

    captured: dict[str, object] = {}

    def fake_dry_run(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(harness, "_dry_run_prepare", fake_dry_run)
    assert harness.main(
        [
            "--out-root",
            str(tmp_path / "paper-main"),
            "--cases",
            "paper-cli",
            "--dry-run",
        ]
    ) == 0
    assert captured["selected_cases"] == ("paper-cli",)


def test_paper_only_selection_has_four_request_cap_and_no_default_gate(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    selected = ("paper-cli",)
    cases = harness._bound_case_specs(
        default_baseline=tmp_path / "unused-default-baseline",
        paper_baseline=tmp_path / "historical-paper-baseline",
        selected_cases=selected,
    )
    assert tuple(cases) == selected
    assert harness.request_budget(
        fresh_default=False,
        fresh_paper=True,
        cases=selected,
    ) == 4
    assert harness._maximum_selected_request_cap(selected) == 4

    choices = {
        "paper-cli": harness.BaselineChoice(
            name="native-pair",
            reuse=False,
            reasons=("fresh source",),
            source_changes=(),
            config_differences=(),
            weights_match=True,
            gpu_match=True,
            runtime_image_match=True,
            rng_evidence_present=True,
        )
    }
    plan = harness._plan_payload(
        cases=cases,
        choices=choices,
        source_fingerprint="b" * 64,
        budget=4,
        environment_checked=True,
        selected_cases=selected,
        request_cap=4,
    )
    assert plan["selected_cases"] == ["paper-cli"]
    assert plan["request_budget"] == 4
    assert plan["hard_request_cap"] == 4
    assert [case["name"] for case in plan["cases"]] == ["paper-cli"]
    assert "paper" in plan["scientific_profile"]
    assert "default" not in plan["scientific_profile"]


def test_merged_plan_reuses_default_and_recreates_only_paper_baseline(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    cases = harness._bound_case_specs(
        default_baseline=tmp_path / "portable-v3-native-single",
        paper_baseline=tmp_path / "old-native-pair",
    )
    choices = {
        "default-cli": harness.BaselineChoice(
            name="native-single",
            reuse=True,
            reasons=(),
            source_changes=(),
            config_differences=(),
            weights_match=True,
            gpu_match=True,
            runtime_image_match=True,
            rng_evidence_present=True,
        ),
        "paper-cli": harness.BaselineChoice(
            name="native-pair",
            reuse=False,
            reasons=("source changed",),
            source_changes=({"path": "astrolabe/run.py"},),
            config_differences=(),
            weights_match=True,
            gpu_match=True,
            runtime_image_match=True,
            rng_evidence_present=True,
        ),
    }

    plan = harness._plan_payload(
        cases=cases,
        choices=choices,
        source_fingerprint="a" * 64,
        budget=harness.request_budget(
            fresh_default=False,
            fresh_paper=True,
        ),
        environment_checked=True,
    )

    assert plan["request_budget"] == 6
    assert plan["hard_request_cap"] == 8
    assert plan["cold_guard"] == {
        "boundary": harness.COLD_GUARD_BOUNDARY,
        "model_requests": 0,
        "required_before_model_requests": True,
    }
    assert plan["cases"][0]["fresh_baseline_requests"] == 0
    assert plan["cases"][0]["fresh_baseline_allowed"] is False
    assert plan["cases"][1]["fresh_baseline_requests"] == 2
    assert plan["cases"][1]["fresh_baseline_allowed"] is True


def test_cut_injection_fires_only_after_raw_prediction_call_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    from astrolabe.design.runtime import Runtime

    case_root = tmp_path / "case"
    (case_root / "instrumentation").mkdir(parents=True)
    (case_root / "instrumentation" / "sitecustomize.py").write_text(
        "",
        encoding="utf-8",
    )
    bootstrap = harness._write_cut_sitecustomize(case_root)
    marker = tmp_path / "marker.json"
    calls = []

    def fake_call(self, named_round, operation, request, *, work_id=None):
        calls.append((named_round, operation, request, work_id))
        return {"round": named_round}

    class ForcedExit(BaseException):
        pass

    def forced_exit(code):
        raise ForcedExit(code)

    monkeypatch.setattr(Runtime, "call", fake_call)
    monkeypatch.setattr(harness.os, "_exit", forced_exit)
    monkeypatch.setenv("ASTROLABE_PUBLIC_CLI_CUT_MARKER", str(marker))
    monkeypatch.setenv("ASTROLABE_PUBLIC_CLI_CUT_BOUNDARY", harness.CUT_BOUNDARY)

    runpy.run_path(str(bootstrap / "sitecustomize.py"))

    assert Runtime.call(  # type: ignore[misc]
        object(),
        "root/finalize_state",
        "operation",
        "request",
    ) == {"round": "root/finalize_state"}
    with pytest.raises(ForcedExit) as captured:
        Runtime.call(  # type: ignore[misc]
            object(),
            harness.CUT_BOUNDARY,
            "operation",
            "request",
        )
    assert captured.value.args == (harness.CUT_EXIT,)
    assert calls == [
        ("root/finalize_state", "operation", "request", None),
        (harness.CUT_BOUNDARY, "operation", "request", None),
    ]
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["task_name"] == harness.TASK_NAME
    assert payload["boundary"] == harness.CUT_BOUNDARY


def test_cold_guard_exits_only_after_runtime_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    from astrolabe.design.runtime import Runtime

    case_root = tmp_path / "cold-case"
    (case_root / "instrumentation").mkdir(parents=True)
    (case_root / "instrumentation" / "sitecustomize.py").write_text(
        "",
        encoding="utf-8",
    )
    bootstrap = harness._write_instrumented_sitecustomize(
        case_root,
        "cold-guard",
    )
    marker = tmp_path / "runtime-marker.json"
    calls = []

    def fake_init(self, *args, **kwargs):
        calls.append(("init", args, kwargs))
        self.root = tmp_path / "runtime"
        self.operations = {
            "evaluator": SimpleNamespace(
                descriptor=SimpleNamespace(
                    identity=SimpleNamespace(digest="evaluator-v1")
                )
            )
        }
        self.scientific_contract = SimpleNamespace(digest="contract-v1")
        self.budgets = {}

    class ForcedExit(BaseException):
        pass

    def forced_exit(code):
        raise ForcedExit(code)

    monkeypatch.setattr(Runtime, "__init__", fake_init)
    monkeypatch.setattr(harness.os, "_exit", forced_exit)
    monkeypatch.setenv(
        "ASTROLABE_PUBLIC_CLI_COLD_GUARD_MARKER",
        str(marker),
    )
    monkeypatch.setenv(
        "ASTROLABE_PUBLIC_CLI_COLD_GUARD_BOUNDARY",
        harness.COLD_GUARD_BOUNDARY,
    )

    runpy.run_path(str(bootstrap / "sitecustomize.py"))

    with pytest.raises(ForcedExit) as captured:
        Runtime()

    assert captured.value.args == (harness.COLD_GUARD_EXIT,)
    assert len(calls) == 1
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["boundary"] == harness.COLD_GUARD_BOUNDARY
    assert payload["runtime_root"] == str(tmp_path / "runtime")
    assert payload["operation_aliases"] == ["evaluator"]
    assert payload["scientific_contract"] == "contract-v1"
    assert payload["operation_identities"] == {
        "evaluator": "evaluator-v1"
    }


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    harness = _load_harness()
    out_root = tmp_path / "attempt_001"
    out_root.mkdir()
    protected = out_root / "result.json"
    protected.write_text('{"status":"old"}\n', encoding="utf-8")
    before = protected.read_bytes()

    with pytest.raises(harness.AcceptanceError, match="refusing to overwrite"):
        harness._reject_existing_output(out_root)

    assert protected.read_bytes() == before
