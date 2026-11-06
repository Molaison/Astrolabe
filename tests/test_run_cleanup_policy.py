from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import run
from astrolabe.run_config_lifecycle import build_persisted_run_config


def test_saved_config_keeps_explicit_losses_used_outside_stage_expression():
    cfg = OmegaConf.create({
        "stages": {"default": {"weighted_score_expression": "plddt"}},
        "loss": {"loss_terms": {
            "report_only": {"_type": "plddt"},
            "hook_score": {"_type": "iptm"},
            "disabled": None,
        }},
    })
    saved = build_persisted_run_config(cfg)
    reloaded = OmegaConf.create(OmegaConf.to_yaml(saved, resolve=False))
    assert set(reloaded.loss.loss_terms) == {"report_only", "hook_score"}
    assert reloaded.loss.loss_terms.hook_score._type == "iptm"


def test_cleanup_keeps_auto_managed_dir_on_failure(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    marker = raw_dir / ".gh_auto_managed"
    marker.write_text("ok\n", encoding="utf-8")
    args = SimpleNamespace(
        _raw_model_results_dir_auto_managed=True,
        raw_model_results_dir=raw_dir,
        _raw_model_results_dir_marker=str(marker),
    )

    run._cleanup_auto_managed_raw_dir(args=args, succeeded=False)
    assert raw_dir.exists()


def test_cleanup_requires_marker_for_auto_managed_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    args = SimpleNamespace(
        _raw_model_results_dir_auto_managed=True,
        raw_model_results_dir=raw_dir,
        _raw_model_results_dir_marker=str(raw_dir / ".gh_auto_managed"),
    )

    with pytest.raises(run.RawModelResultsCleanupError, match="without marker"):
        run._cleanup_auto_managed_raw_dir(args=args, succeeded=True)


def test_cleanup_deletes_auto_managed_dir_with_marker(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    marker = raw_dir / ".gh_auto_managed"
    marker.write_text("ok\n", encoding="utf-8")
    args = SimpleNamespace(
        _raw_model_results_dir_auto_managed=True,
        raw_model_results_dir=raw_dir,
        _raw_model_results_dir_marker=str(marker),
    )

    run._cleanup_auto_managed_raw_dir(args=args, succeeded=True)
    assert not raw_dir.exists()


def test_run_repeats_derives_distinct_seeds_from_explicit_base_seed(monkeypatch):
    cfg = OmegaConf.create(
        {
            "task_name": "demo",
            "repeats": 3,
            "seed": 17,
        }
    )

    seen = []
    monkeypatch.setattr(run, "install_handlers", lambda: None)
    monkeypatch.setattr(run, "stop_requested", lambda: False)
    monkeypatch.setattr(run, "_run_pipeline", lambda run_cfg: seen.append(int(run_cfg.seed)))

    run.run_repeats(cfg)

    assert seen == [17, 18, 19]


def test_run_repeats_isolates_explicit_raw_model_results_dir_by_repeat(
    monkeypatch, tmp_path
):
    raw_root = tmp_path / "raw"
    cfg = OmegaConf.create(
        {
            "task_name": "demo",
            "repeats": 3,
            "seed": 17,
            "raw_model_results_dir": str(raw_root),
        }
    )

    seen = []
    monkeypatch.setattr(run, "install_handlers", lambda: None)
    monkeypatch.setattr(run, "stop_requested", lambda: False)
    monkeypatch.setattr(
        run,
        "_run_pipeline",
        lambda run_cfg: seen.append(str(run_cfg.raw_model_results_dir)),
    )

    run.run_repeats(cfg)

    assert seen == [
        str(raw_root / "repeat_001"),
        str(raw_root / "repeat_002"),
        str(raw_root / "repeat_003"),
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repeats", True),
        ("repeats", 1.5),
        ("seed", True),
        ("seed", 1.5),
    ],
)
def test_run_repeats_rejects_coercible_non_integer_lifecycle_values_before_run(
    monkeypatch, field, value
):
    cfg = {"task_name": "demo", "repeats": 1, "seed": 17}
    cfg[field] = value
    events = []
    monkeypatch.setattr(run, "install_handlers", lambda: events.append("handlers"))
    monkeypatch.setattr(run, "stop_requested", lambda: False)
    monkeypatch.setattr(run, "_run_pipeline", lambda _: events.append("pipeline"))

    with pytest.raises(ValueError, match=field):
        run.run_repeats(OmegaConf.create(cfg))

    assert events == []


def test_run_repeats_rejects_explicit_seed_overflow_before_run(monkeypatch):
    cfg = OmegaConf.create(
        {
            "task_name": "demo",
            "repeats": 2,
            "seed": 2**32 - 1,
        }
    )
    events = []
    monkeypatch.setattr(run, "install_handlers", lambda: events.append("handlers"))
    monkeypatch.setattr(run, "stop_requested", lambda: False)
    monkeypatch.setattr(run, "_run_pipeline", lambda _: events.append("pipeline"))

    with pytest.raises(ValueError, match="seed sequence"):
        run.run_repeats(cfg)

    assert events == []


def test_run_repeats_rejects_resume_with_multiple_repeats_before_setup(monkeypatch):
    cfg = OmegaConf.create(
        {
            "task_name": "demo",
            "repeats": 2,
            "seed": 17,
            "search": {
                "algorithm": "branching",
                "owner": "native_refinement",
                "resume": True,
            },
        }
    )
    events = []
    monkeypatch.setattr(run, "install_handlers", lambda: events.append("handlers"))
    monkeypatch.setattr(run, "plan_repeat_runs", lambda _: events.append("plan"))

    with pytest.raises(ValueError, match="exactly one repeat"):
        run.run_repeats(cfg)

    assert events == []
