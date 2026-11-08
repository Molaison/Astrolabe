from __future__ import annotations

from pathlib import Path
import threading

from omegaconf import OmegaConf
import pytest

from astrolabe import cli, paper
from astrolabe.configuration import (
    SEARCH_FACTORY_CATALOG,
    load_composed_config,
    selected_search_target,
    validate_execution_selection,
)
import run as run_module


@pytest.mark.parametrize(
    "search, error, match",
    [
        ({}, ValueError, "non-empty factory"),
        ({"owner": "isolated_native"}, ValueError, "search.algorithm"),
        ({"algorithm": "branching"}, ValueError, "search.owner"),
        (
            {"algorithm": "", "owner": "isolated_native"},
            ValueError,
            "search.algorithm",
        ),
        (
            {"algorithm": "branching", "owner": "missing"},
            ValueError,
            "Unknown selected search factory",
        ),
    ],
)
def test_explicit_search_must_be_complete_and_registered(
    search,
    error,
    match,
) -> None:
    cfg = OmegaConf.create({"search": search})

    with pytest.raises(error, match=match):
        validate_execution_selection(cfg)


def test_explicit_null_search_selects_only_the_trajectory_pipeline() -> None:
    cfg = OmegaConf.create({"search": None})

    assert selected_search_target(cfg) is None
    assert validate_execution_selection(cfg) is None


def test_search_factory_catalog_separates_generic_native_protein_and_rna() -> None:
    assert SEARCH_FACTORY_CATALOG[
        ("branching", "native_refinement")
    ] == "astrolabe.design.configured_refinement:run_configured_refinement"
    assert SEARCH_FACTORY_CATALOG[
        ("branching", "isolated_native")
    ] == "astrolabe.protein_beam:run_configured_branching"
    assert SEARCH_FACTORY_CATALOG[
        ("branching", "archived_refinement")
    ] == "astrolabe.rna_beam:run_configured_refinement"


@pytest.mark.parametrize(
    "owner",
    ["native_refinement", "isolated_native", "archived_refinement"],
)
def test_explicit_search_rejects_nonempty_legacy_beam_search(owner: str) -> None:
    cfg = OmegaConf.create(
        {
            "search": {"algorithm": "branching", "owner": owner},
            "beam_search": {"initialization": "archived"},
        }
    )

    with pytest.raises(ValueError, match="Legacy beam_search cannot be combined"):
        validate_execution_selection(cfg)


def test_isolated_native_owns_protein_beam_execution_settings() -> None:
    cfg = OmegaConf.create(
        {
            "search": {"algorithm": "branching", "owner": "isolated_native"},
            "protein_beam": {"width": 4, "max_evaluations": 7},
        }
    )

    assert validate_execution_selection(cfg) == (
        "branching",
        "isolated_native",
    )
    assert cfg.protein_beam.width == 4
    assert cfg.protein_beam.max_evaluations == 7


def test_non_isolated_search_rejects_effective_protein_beam_execution() -> None:
    cfg = OmegaConf.create(
        {
            "search": {"algorithm": "branching", "owner": "archived_refinement"},
            "protein_beam": {"width": 4},
        }
    )

    with pytest.raises(ValueError, match="protein_beam.*conflict.*archived_refinement"):
        validate_execution_selection(cfg)


def test_shared_protein_beam_capacity_fields_remain_harmless_with_search() -> None:
    cfg = OmegaConf.create(
        {
            "search": {"algorithm": "branching", "owner": "archived_refinement"},
            "protein_beam": {"workers": 3, "in_flight": 3},
        }
    )

    assert validate_execution_selection(cfg) == (
        "branching",
        "archived_refinement",
    )


def test_load_composed_config_overlays_repository_base_for_standalone_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "standalone.yaml"
    path.write_text("task_name: local-task\nseqs:\n  A: ACDE\n")

    cfg = load_composed_config(path)

    assert cfg.task_name == "local-task"
    assert cfg.seqs.A == "ACDE"
    assert cfg.steps >= 1


def test_load_composed_config_preserves_hydra_defaults_composition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "local_defaults.yaml"
    path.write_text(
        "defaults:\n"
        "  - base\n"
        "  - _self_\n"
        "task_name: defaults-task\n"
    )

    cfg = load_composed_config(path)

    assert cfg.task_name == "defaults-task"
    assert cfg.steps >= 1


def test_shared_protein_beam_defaults_do_not_force_beam_execution() -> None:
    cfg = OmegaConf.create(
        {
            "steps": 7,
            "protein_beam": {"workers": 3, "in_flight": 3},
        }
    )

    assert selected_search_target(cfg) is None
    assert validate_execution_selection(cfg) is None


def test_obvious_protein_beam_configuration_requires_explicit_search() -> None:
    cfg = OmegaConf.create(
        {
            "protein_beam": {
                "width": 4,
                "proposals_per_parent": 6,
                "max_evaluations": 7,
            }
        }
    )

    with pytest.raises(ValueError, match="Protein-beam.*search.algorithm"):
        validate_execution_selection(cfg)


def test_legacy_engine_and_beam_search_cannot_fall_back_to_trajectory() -> None:
    with pytest.raises(
        ValueError,
        match="deprecated.*search.algorithm",
    ):
        validate_execution_selection(
            OmegaConf.create(
                {"paper_protocol": {"engine": "rna_beam"}}
            )
        )

    with pytest.raises(ValueError, match="Legacy beam_search"):
        validate_execution_selection(
            OmegaConf.create(
                {
                    "beam_search": {
                        "initialization": "archived",
                        "archived_starts": [{"sequence": "ACGU"}],
                    }
                }
            )
        )

    assert validate_execution_selection(
        OmegaConf.create({"paper_protocol": {"engine": "trajectory"}})
    ) is None
    assert validate_execution_selection(
        OmegaConf.create({"beam_search": {}})
    ) is None


def test_old_rna_measurement_requires_an_explicit_owner_migration() -> None:
    cfg = OmegaConf.create(
        {
            "search": {
                "algorithm": "branching",
                "owner": "native_refinement",
                "measurement": "confidence",
            }
        }
    )

    with pytest.raises(ValueError, match="archived_refinement"):
        validate_execution_selection(cfg)

    migrated = OmegaConf.create(
        {
            "search": {
                "algorithm": "branching",
                "owner": "archived_refinement",
                "measurement": "confidence",
            }
        }
    )
    assert validate_execution_selection(migrated) == (
        "branching",
        "archived_refinement",
    )


def test_paper_null_override_selects_trajectory_without_a_profile_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dump = tmp_path / "selected.yaml"
    ns = cli._build_parser().parse_args(
        [
            "paper",
            "run",
            "rna-oligomer-4",
            "--overrides",
            "search=null",
            "--dump-config",
            str(dump),
            "--dry-run",
            "--config-env",
            "env",
        ]
    )
    monkeypatch.setattr(
        paper,
        "_compose",
        lambda _ns: OmegaConf.create(
            {
                "paper_protocol": {},
                "seqs": {"A": "ACGU"},
                "search": {
                    "algorithm": "branching",
                    "owner": "archived_refinement",
                    "measurement": "confidence",
                },
            }
        ),
    )
    monkeypatch.setattr(
        cli,
        "_run_from_config",
        lambda *args, **kwargs: pytest.fail("invalid route reached execution"),
    )

    assert paper._run(ns) == 0

    assert OmegaConf.load(dump).get("search") is None


def test_paper_null_override_rejects_retained_legacy_marker_before_dump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dump = tmp_path / "selected.yaml"
    ns = cli._build_parser().parse_args(
        [
            "paper",
            "run",
            "rna-oligomer-4",
            "--overrides",
            "search=null",
            "--dump-config",
            str(dump),
            "--dry-run",
            "--config-env",
            "env",
        ]
    )
    monkeypatch.setattr(
        paper,
        "_compose",
        lambda _ns: OmegaConf.create(
            {
                "paper_protocol": {"engine": "rna_beam"},
                "seqs": {"A": "ACGU"},
                "search": {
                    "algorithm": "branching",
                    "owner": "archived_refinement",
                    "measurement": "confidence",
                },
            }
        ),
    )
    monkeypatch.setattr(
        cli,
        "_run_from_config",
        lambda *args, **kwargs: pytest.fail("invalid route reached execution"),
    )

    with pytest.raises(ValueError, match="deprecated.*search.algorithm"):
        paper._run(ns)

    assert not dump.exists()


def test_run_repeats_validates_before_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr("astrolabe.startup_validation.validate_startup_config", lambda *a, **k: None)

    def reject(_cfg):
        events.append("validate")
        raise ValueError("invalid execution selection")

    monkeypatch.setattr(run_module, "validate_execution_selection", reject)
    monkeypatch.setattr(
        run_module,
        "plan_repeat_runs",
        lambda _cfg: events.append("plan") or (),
    )

    with pytest.raises(ValueError, match="invalid execution selection"):
        run_module.run_repeats(OmegaConf.create({}))

    assert events == ["validate"]


def test_run_repeats_revalidates_each_materialized_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from astrolabe.design.repeats import RepeatRun
    monkeypatch.setattr("astrolabe.startup_validation.validate_startup_config", lambda *a, **k: None)

    events: list[str] = []

    def validate(_cfg):
        events.append("validate")
        if events.count("validate") == 2:
            raise ValueError("materialized execution selection is invalid")
        return None

    repeat = RepeatRun(index=0, total=1, seed=7)
    monkeypatch.setattr(run_module, "validate_execution_selection", validate)
    monkeypatch.setattr(
        run_module,
        "plan_repeat_runs",
        lambda _cfg: events.append("plan") or (repeat,),
    )
    monkeypatch.setattr(
        run_module,
        "materialize_repeat",
        lambda cfg, _repeat: events.append("materialize") or cfg,
    )
    monkeypatch.setattr(run_module, "install_handlers", lambda: None)
    monkeypatch.setattr(
        run_module,
        "dispatch_selected_search",
        lambda *_args, **_kwargs: pytest.fail("invalid selection was dispatched"),
    )
    monkeypatch.setattr(
        run_module,
        "_run_pipeline",
        lambda *_args, **_kwargs: pytest.fail("invalid selection reached the pipeline"),
    )

    with pytest.raises(ValueError, match="materialized execution selection"):
        run_module.run_repeats(OmegaConf.create({}))

    assert events == ["validate", "plan", "materialize", "validate"]


def test_run_repeats_holds_scientific_owner_through_setup_and_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext

    from astrolabe import model_execution
    from astrolabe.design.owner import SCIENTIFIC_OWNER
    from astrolabe.design.repeats import RepeatRun

    monkeypatch.setattr("astrolabe.startup_validation.validate_startup_config", lambda *a, **k: None)
    reset_entered = threading.Event()
    release_reset = threading.Event()
    pipeline_entered = threading.Event()
    release_pipeline = threading.Event()
    errors: list[BaseException] = []
    repeat = RepeatRun(index=0, total=1, seed=7)

    def reset() -> None:
        reset_entered.set()
        if not release_reset.wait(5):
            raise TimeoutError("test did not release reset")

    def run_pipeline(_cfg) -> None:
        pipeline_entered.set()
        if not release_pipeline.wait(5):
            raise TimeoutError("test did not release pipeline")

    def assert_owner_unavailable() -> None:
        acquired = SCIENTIFIC_OWNER.acquire(blocking=False)
        if acquired:
            SCIENTIFIC_OWNER.release()
        assert not acquired

    monkeypatch.setattr(run_module, "validate_execution_selection", lambda _cfg: None)
    monkeypatch.setattr(run_module, "plan_repeat_runs", lambda _cfg: (repeat,))
    monkeypatch.setattr(run_module, "materialize_repeat", lambda cfg, _repeat: cfg)
    monkeypatch.setattr(run_module, "install_handlers", lambda: None)
    monkeypatch.setattr(
        run_module,
        "dispatch_selected_search",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(run_module, "_run_pipeline", run_pipeline)
    monkeypatch.setattr(run_module, "stop_requested", lambda: False)
    monkeypatch.setattr(run_module.global_config, "reset", reset)
    monkeypatch.setattr(model_execution, "model_run_scope", nullcontext)

    def run_repeats() -> None:
        try:
            run_module.run_repeats(OmegaConf.create({}))
        except BaseException as error:
            errors.append(error)
        finally:
            release_reset.set()
            release_pipeline.set()

    worker = threading.Thread(target=run_repeats)
    worker.start()
    assert reset_entered.wait(5)
    assert_owner_unavailable()
    release_reset.set()
    assert pipeline_entered.wait(5)
    assert_owner_unavailable()
    release_pipeline.set()
    worker.join(5)

    assert not worker.is_alive()
    assert not errors
    assert SCIENTIFIC_OWNER.acquire(blocking=False)
    SCIENTIFIC_OWNER.release()
