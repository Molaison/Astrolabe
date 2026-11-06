from __future__ import annotations

from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest

import astrolabe.configuration as configuration
import astrolabe.design as design
import astrolabe.protein_beam as beam
from astrolabe.design.repeats import RepeatRun


def _configured_cfg(tmp_path: Path, **execution_overrides):
    execution = {
        "socket_path": str(tmp_path / "af3.sock"),
        "workers": 4,
        "in_flight": 3,
        "resume": False,
        "retry_interrupted": False,
    }
    execution.update(execution_overrides)
    return OmegaConf.create(
        {
            "task_name": "routing-test",
            "seed": 11,
            "repeats": 1,
            "scientific_option": "first",
            "out_dir": str(tmp_path / "configured"),
            "raw_model_results_dir": None,
            "protein_beam": {
                "workers": 2,
                "in_flight": 2,
                "af3_call_budget": {
                    "ledger": "${out_dir}/af3_calls.json",
                    "scope": "${task_name}",
                },
            },
            "evaluator": {"alphafold3": {"service_socket": None}},
            "search": {
                "algorithm": "branching",
                "owner": "isolated_native",
                "execution": execution,
            },
        }
    )


def test_load_config_is_the_shared_composed_config_loader() -> None:
    assert beam.load_config is configuration.load_composed_config


def test_configured_branching_forwards_exact_execution_without_starting_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _configured_cfg(tmp_path)
    output = tmp_path / "repeat-output"
    repeat = RepeatRun(index=0, total=1, seed=29, output=output)
    calls = []

    def run_beam(value, run_output, socket_path, **kwargs):
        calls.append((value, Path(run_output), socket_path, kwargs))
        return {"status": "routed"}

    def forbidden_service(*args, **kwargs):
        raise AssertionError("configured routing must not start an AF3 service")

    monkeypatch.setattr(beam, "run_beam", run_beam)
    monkeypatch.setattr(
        "astrolabe.evaluators.af3_lifecycle.OwnedAF3",
        forbidden_service,
        raising=False,
    )

    result = beam.run_configured_branching(cfg, repeat)

    assert result == {"status": "routed"}
    value, run_output, socket_path, kwargs = calls[0]
    assert run_output == output.resolve()
    assert socket_path == str((tmp_path / "af3.sock").resolve())
    assert kwargs == {
        "workers": 4,
        "in_flight": 3,
        "resume": False,
        "retry_interrupted": False,
    }
    assert value.seed == 29
    assert value.repeats == 1
    assert value.out_dir == str(output.resolve())
    assert value.raw_model_results_dir is None


@pytest.mark.parametrize(
    ("field", "value"),
    [("algorithm", "other"), ("owner", "rna_only")],
)
def test_configured_branching_rejects_explicit_selection_mismatch_before_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    cfg = _configured_cfg(tmp_path)
    cfg.search[field] = value
    repeat = RepeatRun(index=0, total=1, seed=29, output=tmp_path / "out")
    monkeypatch.setattr(
        beam,
        "run_beam",
        lambda *args, **kwargs: pytest.fail("mismatched search reached run_beam"),
    )

    with pytest.raises(ValueError, match=field):
        beam.run_configured_branching(cfg, repeat)


@pytest.mark.parametrize("field", ["algorithm", "owner"])
def test_configured_branching_rejects_partial_search_selection_before_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
) -> None:
    cfg = _configured_cfg(tmp_path)
    del cfg.search[field]
    repeat = RepeatRun(index=0, total=1, seed=29, output=tmp_path / "out")
    monkeypatch.setattr(
        beam,
        "run_beam",
        lambda *args, **kwargs: pytest.fail("partial search reached run_beam"),
    )

    with pytest.raises(ValueError, match=rf"search\.{field}.*required"):
        beam.run_configured_branching(cfg, repeat)


def test_configured_branching_rejects_unknown_search_fields_before_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _configured_cfg(tmp_path)
    cfg.search.width = 3
    repeat = RepeatRun(index=0, total=1, seed=29, output=tmp_path / "out")
    monkeypatch.setattr(
        beam,
        "run_beam",
        lambda *args, **kwargs: pytest.fail("unknown search field reached run_beam"),
    )

    with pytest.raises(ValueError, match="unknown=.*width"):
        beam.run_configured_branching(cfg, repeat)


@pytest.mark.parametrize("summary_format", ["csv", "text"])
def test_configured_branching_uses_the_central_summary_format_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    summary_format: str,
) -> None:
    cfg = _configured_cfg(tmp_path)
    cfg.search.summary_format = summary_format
    repeat = RepeatRun(index=0, total=1, seed=29, output=tmp_path / "out")
    monkeypatch.setattr(
        beam,
        "run_beam",
        lambda *args, **kwargs: pytest.fail("invalid summary format reached run_beam"),
    )

    with pytest.raises(ValueError, match="summary_format"):
        beam.run_configured_branching(cfg, repeat)


def test_configured_branching_rejects_legacy_beam_conflict_before_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _configured_cfg(tmp_path)
    cfg.beam_search = {"algorithm": "legacy"}
    repeat = RepeatRun(index=0, total=1, seed=29, output=tmp_path / "out")
    monkeypatch.setattr(
        beam,
        "run_beam",
        lambda *args, **kwargs: pytest.fail("conflicting beam reached run_beam"),
    )

    with pytest.raises(ValueError, match="Legacy beam_search"):
        beam.run_configured_branching(cfg, repeat)


def test_configured_branching_requires_socket_before_output_or_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _configured_cfg(tmp_path, socket_path=None)
    output = tmp_path / "must-not-exist"
    repeat = RepeatRun(index=0, total=1, seed=29, output=output)
    monkeypatch.setattr(
        beam,
        "run_beam",
        lambda *args, **kwargs: pytest.fail("missing socket reached run_beam"),
    )

    with pytest.raises(ValueError, match="socket_path"):
        beam.run_configured_branching(cfg, repeat)
    assert not output.exists()


def test_dedicated_cli_uses_canonical_execution_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _configured_cfg(tmp_path)
    cfg.protein_beam.workers = 9
    cfg.protein_beam.in_flight = 9
    cfg.evaluator.alphafold3.service_socket = str(tmp_path / "fallback.sock")
    calls = []

    monkeypatch.setattr(beam, "load_config", lambda path: cfg)
    monkeypatch.setattr(
        beam,
        "run_beam",
        lambda value, output, socket, **kwargs: calls.append(
            (value, Path(output), socket, kwargs)
        )
        or {
            "status": "complete",
            "records": [],
            "budget_complete": True,
            "wall_s": 0.0,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "protein_beam",
            "--config",
            "unused.yaml",
            "--out",
            str(tmp_path / "cli-output"),
        ],
    )

    assert beam.main() == 0
    value, output, socket, kwargs = calls[0]
    assert output == (tmp_path / "cli-output").resolve()
    assert socket == str((tmp_path / "af3.sock").resolve())
    assert kwargs == {
        "workers": 4,
        "in_flight": 3,
        "resume": False,
        "retry_interrupted": False,
    }
    assert value.protein_beam.af3_call_budget.ledger == str(
        output / "af3_calls.json"
    )


def test_run_beam_saves_canonical_config_for_configured_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "beam-output"
    cfg = OmegaConf.create(
        {
            "task_name": "save-test",
            "seed": 101,
            "repeats": 1,
            "out_dir": str(output),
            "raw_model_results_dir": None,
            "repeat_plan": {"total_runs": 3},
            "seqs": {"A": "ACDE", "B": "+FGHI"},
            "search": {
                "algorithm": "branching",
                "owner": "isolated_native",
                "execution": {
                    "socket_path": None,
                    "workers": 4,
                    "in_flight": 3,
                    "resume": False,
                    "retry_interrupted": False,
                },
                "summary_format": "json",
            },
            "protein_beam": {
                "width": 1,
                "proposals_per_parent": 1,
                "max_evaluations": 1,
                "selection": "weighted",
                "holes_ranking": "raw_sum",
            },
        }
    )
    socket_path = tmp_path / "owner.sock"
    contract_inputs = []

    monkeypatch.setattr(beam, "validate_config", lambda value: "objective")
    monkeypatch.setattr(
        "astrolabe.protein_beam_contract.scientific_contract",
        lambda value: (
            contract_inputs.append(
                OmegaConf.create(OmegaConf.to_container(value, resolve=False))
            )
            or {"config_yaml": "scientific"},
            "digest",
        ),
    )
    monkeypatch.setattr(
        "astrolabe.protein_beam_calibration.calibration_enabled",
        lambda value: False,
    )

    class StopAfterSave(RuntimeError):
        pass

    def stop_after_save(*args, **kwargs):
        raise StopAfterSave

    monkeypatch.setattr(design, "Runtime", stop_after_save)
    with pytest.raises(StopAfterSave):
        beam.run_beam(
            cfg,
            output,
            socket_path,
            workers=4,
            in_flight=3,
            resume=False,
            retry_interrupted=False,
        )

    saved = OmegaConf.load(output / "config.yaml")
    assert saved.seed == 101
    assert saved.repeats == 1
    assert saved.out_dir == str(output.resolve())
    assert saved.raw_model_results_dir is None
    assert "repeat_plan" not in saved
    assert saved.search.algorithm == "branching"
    assert saved.search.owner == "isolated_native"
    assert saved.search.summary_format == "json"
    assert OmegaConf.to_container(saved.search.execution, resolve=True) == {
        "socket_path": str(socket_path.resolve()),
        "workers": 4,
        "in_flight": 3,
        "resume": False,
        "retry_interrupted": False,
    }
    assert "search" not in contract_inputs[0]
    assert contract_inputs[0].seed == 101
    assert contract_inputs[0].out_dir == str(output.resolve())
    assert contract_inputs[0].repeats == 1

    replay = OmegaConf.create(OmegaConf.to_container(saved, resolve=False))
    repeat = RepeatRun(
        index=0,
        total=1,
        seed=saved.seed,
        output=Path(replay.out_dir),
    )
    replay_calls = []

    def fake_run_beam(value, run_output, socket, **kwargs):
        replay_calls.append((value, Path(run_output), socket, kwargs))
        return {"status": "complete"}

    monkeypatch.setattr(beam, "run_beam", fake_run_beam)
    assert configuration.dispatch_selected_search(replay, repeat) is True
    value, run_output, socket, kwargs = replay_calls[0]
    assert run_output == output.resolve()
    assert socket == str(socket_path.resolve())
    assert kwargs["workers"] == 4
    assert kwargs["in_flight"] == 3
    assert OmegaConf.to_container(
        beam._scientific_contract_config(value), resolve=False
    ) == OmegaConf.to_container(contract_inputs[0], resolve=False)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("algorithm", "other", "search.algorithm"),
        ("owner", "rna_only", "search.owner"),
        ("policy", {"mode": "legacy"}, "unknown=.*policy"),
        ("width", 3, "unknown=.*width"),
    ],
)
def test_validate_config_rejects_invalid_search_schema(
    tmp_path: Path,
    field: str,
    value,
    match: str,
) -> None:
    cfg = _configured_cfg(tmp_path)
    cfg.search[field] = value

    with pytest.raises(ValueError, match=match):
        beam.validate_config(cfg)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("algorithm", "other", "search.algorithm"),
        ("owner", "rna_only", "search.owner"),
        ("policy", {"mode": "legacy"}, "unknown=.*policy"),
        ("width", 3, "unknown=.*width"),
    ],
)
def test_run_beam_rejects_invalid_search_before_creating_output(
    tmp_path: Path,
    field: str,
    value,
    match: str,
) -> None:
    cfg = _configured_cfg(tmp_path)
    cfg.search[field] = value
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match=match):
        beam.run_beam(
            cfg,
            output,
            tmp_path / "actual-owner.sock",
            workers=4,
            in_flight=3,
        )
    assert not output.exists()


def test_run_beam_rejects_invalid_execution_map_before_creating_output(
    tmp_path: Path,
) -> None:
    cfg = beam.load_config(
        str(Path(__file__).parent / "data/protein_beam/02_fixture.yaml")
    )
    cfg.search = _configured_cfg(tmp_path).search
    del cfg.search.execution.in_flight
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="missing=.*in_flight"):
        beam.run_beam(
            cfg,
            output,
            tmp_path / "actual-owner.sock",
            workers=4,
            in_flight=3,
        )
    assert not output.exists()


def test_scientific_contract_input_ignores_replay_search_and_runtime_controls(
    tmp_path: Path,
) -> None:
    cfg = _configured_cfg(tmp_path)
    cfg.protein_beam.workers = 2
    cfg.protein_beam.in_flight = 1

    first = beam._scientific_contract_config(cfg)
    second_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    second_cfg.search.execution.socket_path = str(tmp_path / "other-owner.sock")
    second_cfg.search.execution.workers = 8
    second_cfg.search.execution.in_flight = 5
    second_cfg.search.execution.resume = True
    second_cfg.search.execution.retry_interrupted = True
    second = beam._scientific_contract_config(second_cfg)

    assert OmegaConf.to_container(first, resolve=False) == OmegaConf.to_container(
        second, resolve=False
    )
    assert "search" not in first
    assert first.protein_beam.workers == 2
    assert first.protein_beam.in_flight == 1
    assert first.out_dir == str(tmp_path / "configured")
    assert first.raw_model_results_dir is None

    changed = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    changed.scientific_option = "second"
    third = beam._scientific_contract_config(changed)
    assert OmegaConf.to_container(third, resolve=False) != OmegaConf.to_container(
        first, resolve=False
    )


def test_historical_markerless_output_requires_explicit_dedicated_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "historical"
    output.mkdir()
    (output / "config.yaml").write_text("task_name: historical\n")
    (output / "beam_state.json").write_text("{}\n")

    cfg = _configured_cfg(tmp_path)
    repeat = RepeatRun(index=0, total=1, seed=29, output=output)
    calls = []

    monkeypatch.setattr(
        beam,
        "run_beam",
        lambda value, run_output, socket, **kwargs: calls.append(
            (value, Path(run_output), socket, kwargs)
        )
        or {"status": "resumed"},
    )

    with pytest.raises(FileExistsError, match="Historical.*no canonical"):
        beam.run_configured_branching(cfg, repeat)
    assert calls == []

    cfg.search.execution.resume = True
    assert beam.run_configured_branching(cfg, repeat) == {"status": "resumed"}
    value, run_output, _socket, kwargs = calls[0]
    assert run_output == output.resolve()
    assert kwargs["resume"] is True
    assert value.task_name == "routing-test"
