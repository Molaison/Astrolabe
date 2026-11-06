from __future__ import annotations

from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf

from astrolabe import Astrolabe, app
from astrolabe.design.composition import DesignRun
from astrolabe.design.runtime import Runtime
from astrolabe.design.store import DataStore, JSONCodec
from astrolabe.design.successor_search import AcceptedSuccessorPolicy
from astrolabe.design.types import (
    Assessment,
    Assignment,
    Candidate,
    Continuation,
    Evidence,
    InputManifest,
    Polymer,
    RunResult,
    Scored,
    TrialOutcome,
    key_for,
    make_candidate,
    make_system,
)


def test_load_run_module_does_not_depend_on_cwd_or_repo_root_on_syspath(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)

    module = app._load_run_module()

    # The runner is resolved from the package location, not from the cwd or sys.path.
    assert Path(module.__file__).resolve() == app.run_script_path().resolve()
    assert Path(module.__file__).name == "run.py"
    assert hasattr(module, "_run_pipeline")
    assert sys.modules[module.__name__] is module


@pytest.mark.parametrize("public_api", [False, True])
def test_run_from_cfg_uses_canonical_repeat_lifecycle(
    monkeypatch, tmp_path: Path, capsys, public_api
) -> None:
    run = app._load_run_module()
    raw_root = tmp_path / "raw"
    monkeypatch.delenv("GH_WAVE1_UNUSED_MISSING", raising=False)
    cfg = OmegaConf.create(
        {
            "task_name": "demo",
            "count": 4,
            "repeats": "${count}",
            "seed": 17,
            "raw_model_results_dir": str(raw_root),
            "unused": "${oc.env:GH_WAVE1_UNUSED_MISSING}",
        }
    )
    seen = []

    monkeypatch.setattr(run, "install_handlers", lambda: None)
    monkeypatch.setattr(run, "stop_requested", lambda: len(seen) == 3)
    monkeypatch.setattr(
        run,
        "_run_pipeline",
        lambda run_cfg: seen.append(
            (int(run_cfg.seed), Path(run_cfg.raw_model_results_dir))
        ),
    )

    if public_api:
        application = Astrolabe(cfg)
        assert application.config is cfg
        application.run()
    else:
        app.run_from_cfg(cfg)

    assert seen == [
        (17, raw_root / "repeat_001"),
        (18, raw_root / "repeat_002"),
        (19, raw_root / "repeat_003"),
    ]
    assert "exiting after repeat 3/4." in capsys.readouterr().err


def _successor_scored(candidate: Candidate, label: str, value: float) -> Scored:
    inputs = InputManifest(key_for(("app-successor-inputs/v1",)), ())
    evidence = Evidence(
        identity=key_for(("app-successor-evidence/v1", candidate.identity)),
        evaluated_candidate=candidate.identity,
        artifacts=(),
        measurements=(),
        inputs=inputs,
    )
    return Scored(
        candidate=candidate,
        evidence=evidence,
        assessment=Assessment(
            evidence=evidence.identity,
            objective_revision=key_for(("app-successor-objective/v1",)),
            history_revision=key_for(("app-successor-history/v1",)),
            values=(value,),
            gates=(),
            inputs=inputs,
        ),
    )


def _composition_fixture(tmp_path: Path):
    data = DataStore(tmp_path / "composition-data")
    topology = data.put({"bonds": []}, JSONCodec())
    system = make_system(("shared",), (), topology, ())
    initial = make_candidate(
        system,
        (Assignment("shared", Polymer("protein", ("G", "G", "G"), ())),),
    )
    successor_candidate = make_candidate(
        system,
        (Assignment("shared", Polymer("protein", ("G", "G", "A"), ())),),
    )
    root = _successor_scored(initial, "root", 10.0)
    successor = _successor_scored(
        successor_candidate,
        "accepted-lower",
        -1.0,
    )

    class Trial:
        requirements = ()
        fail_fast = True

        def __init__(self):
            self.identity = key_for(("app-continuation-trial/v1",))

        def root(self, candidate, runtime):
            assert candidate == initial
            return root

        def perform(self, work, runtime):
            assert work.stage == "advance"
            return TrialOutcome(
                work=work.identity,
                status="completed",
                proposals=(),
                children=(),
                attempts=(),
                continuation=Continuation(successor),
            )

    return data, system, initial, Trial()


def test_astrolabe_design_run_returns_result_and_keeps_caller_runtime_open(
    tmp_path: Path,
) -> None:
    data, system, initial, trial = _composition_fixture(tmp_path)
    policy = AcceptedSuccessorPolicy(rounds=1, seed=31)
    with Runtime(root=tmp_path / "runtime", data=data) as runtime:
        result = Astrolabe(
            DesignRun(
                system=system,
                initial=initial,
                trial=trial,
                policy=policy,
                runtime=runtime,
                capacity=2,
            )
        ).run()

        assert isinstance(result, RunResult)
        assert result.selected[0].assessment.values == (-1.0,)
        assert runtime._closed is False
        assert runtime.load("run/contract") is not None
    assert runtime._closed is True


def test_astrolabe_design_run_delegates_to_coordinator_with_capacity(
    monkeypatch, tmp_path: Path
) -> None:
    data, system, initial, trial = _composition_fixture(tmp_path)
    policy = AcceptedSuccessorPolicy(rounds=1, seed=7)
    runtime = Runtime()
    sentinel = object()
    called = {}

    from astrolabe.design import coordinator

    def spy(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(coordinator, "run", spy)
    monkeypatch.setattr(
        app,
        "_load_run_module",
        lambda: (_ for _ in ()).throw(
            AssertionError("legacy runner must not load")
        ),
    )

    composition = DesignRun(
        system=system,
        initial=initial,
        trial=trial,
        policy=policy,
        runtime=runtime,
        capacity=4,
    )
    result = Astrolabe(composition).run()

    assert result is sentinel
    assert called["args"] == (system, initial, trial, policy, runtime)
    assert called["kwargs"] == {"capacity": 4}
