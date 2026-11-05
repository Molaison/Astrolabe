from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

import run as run_module
from astrolabe.core import global_config
from astrolabe.core.protomer_config_parser import Protomer, SegmentManager
from astrolabe.design import coordinator
from astrolabe.design import native_flow
from astrolabe.design.legacy import translate_legacy_state
from astrolabe.design.process_resources import ProcessReference, ProcessResources
from astrolabe.design.runtime import OutcomeUnknown
from astrolabe.design.store import DataStore
from astrolabe.design.types import History, key_for
from astrolabe.evaluators.utils import ComplexPoses, StandardScores
from astrolabe.losses.base import LossEngine
from astrolabe.stage_manager import StageManager


@pytest.fixture(autouse=True)
def clean_default_coordinator_globals():
    global_config.reset()
    global_config.init_const("segments", {})
    global_config.init_const("protomers", {})
    global_config.init_const("mutable_protomers", {})
    global_config.init_const("named_files", {})
    global_config.init_const("segment_manager", SegmentManager())
    global_config.init_const("use_metropolis_acceptance", False)
    yield
    global_config.reset()


def _shared_segment_state(tmp_path: Path):
    first = Protomer("A", "AC@shared", "p")
    second = Protomer("B", "shared", "p")
    global_config.protomers.update({"A": first, "B": second})
    global_config.mutable_protomers["A"] = first
    shared = global_config.segments["A_1"]
    assert global_config.segments["shared"] is shared
    assert global_config.segments["B_1"] is shared

    loss = LossEngine({"loss_terms": {}})
    global_config.init_const("loss_computer", loss)
    global_config.init_const("out_dir", tmp_path)
    stages = StageManager(
        {
            "default": {
                "oligo": ["AB"],
                "mutation_num": 1,
                "weighted_score_expression": "plddt",
                "evaluator": "Eval",
                "mutators": ["Mutate"],
            }
        },
        loss_computer=loss,
    )
    global_config.init_const("stage_manager", stages)
    args = SimpleNamespace(
        steps=2,
        seed=17,
        out_dir=tmp_path,
        task_name="default-coordinator-smoke",
        mutator=None,
        logging=None,
        _raw_model_results_dir_auto_managed=False,
    )
    return shared, loss, stages, args


def _scores(tmp_path: Path, name: str, value: float) -> StandardScores:
    path = tmp_path / f"{name}.cif"
    path.write_text(f"data_{name}\n# synthetic structure\n", encoding="utf-8")
    return StandardScores(
        names=["AB"],
        plddt={
            "AB": [np.full(4, value, dtype=np.float32)],
        },
        complex_paths={"AB": path},
    )


def test_default_runner_uses_one_coordinator_and_accepts_exact_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared, loss, stages, args = _shared_segment_state(tmp_path)
    initial_scores = _scores(tmp_path, "initial", 0.25)
    successor_scores = _scores(tmp_path, "successor", 0.05)
    events: list[str] = []
    observed: dict[str, object] = {}
    decisions = SimpleNamespace(scores={0: 0.25}, last_accepted=0)

    def initial_loader():
        events.append("initial")
        return initial_scores

    def initialize(args, loss_computer, initial_result, stage_manager, runtime_config):
        events.append("initialize")
        assert initial_result is initial_scores
        return decisions, initial_result

    class _Evaluator:
        def evaluate(self, inputs, runtime_config):
            events.append(f"evaluate:{runtime_config['step']}")
            return successor_scores

    evaluator = _Evaluator()

    def mutate(segment):
        events.append("mutate")
        segment.sequence = "AG"

    def decide(step):
        events.append(f"decision:{step}")
        decisions.scores[step] = 0.05
        decisions.last_accepted = step
        return True

    def advance(
        args,
        loss_computer,
        current,
        stage_manager,
        runtime_config,
        decisions_arg,
        step,
    ):
        assert current is initial_scores
        assert decisions_arg is decisions
        native_flow.native_call("mutation", mutate, shared)
        result = native_flow.native_call(
            "prediction",
            evaluator.evaluate,
            {"AB": "AC"},
            {"step": step},
        )
        native_flow.native_call("decision", decide, step)
        return result, True, True

    monkeypatch.setattr(run_module, "_initialize_optimization_state", initialize)
    monkeypatch.setattr(run_module, "_advance_optimization_state", advance)

    original_coordinator_run = coordinator.run

    def event_spy(system, initial, trial, policy, runtime, *, capacity=1):
        observed["system"] = system
        observed["initial"] = initial
        observed["trial"] = trial
        observed["policy"] = policy
        try:
            return original_coordinator_run(
                system,
                initial,
                trial,
                policy,
                runtime,
                capacity=capacity,
            )
        finally:
            observed["attempts"] = runtime.attempts()

    monkeypatch.setattr(native_flow.coordinator, "run", event_spy)

    result = run_module.run_optimization_loop(
        args,
        loss,
        None,
        stages,
        {},
        initial_loader=initial_loader,
    )

    assert result is successor_scores
    assert events == ["initial", "initialize", "mutate", "evaluate:1", "decision:1"]
    trial = observed["trial"]
    assert isinstance(trial, native_flow.NativeFlow)
    assert trial.system.variables == ("A_1",)
    assert len(trial.system.occurrences) == 2
    assert trial.current is successor_scores
    assert trial.decisions.last_accepted == 1

    attempts = observed["attempts"]
    assert [attempt.key.named_round for attempt in attempts] == [
        "initial/0",
        "prepare/0",
        "mutation/0",
        "prediction/0",
        "decision/0",
    ]
    assert all(attempt.status == "completed" for attempt in attempts)

    execution = (tmp_path / "execution.json").read_text(encoding="utf-8")
    assert '"runner": "astrolabe.design.coordinator"' in execution
    assert '"recovery": "process_lifetime"' in execution


def test_run_pipeline_enters_the_single_native_coordinator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, loss, stages, args = _shared_segment_state(tmp_path)
    args.steps = 1
    global_config.init_const("evaluators", {"Eval": object()})
    initial_scores = _scores(tmp_path, "pipeline-root", 0.6)
    decisions = SimpleNamespace(scores={0: 0.6}, last_accepted=0)
    calls = []

    monkeypatch.setattr(run_module, "parse_arguments", lambda cfg: args)
    monkeypatch.setattr(run_module, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        run_module,
        "initialize_pipeline_components",
        lambda parsed: (loss, stages, {"AB": "input"}),
    )
    monkeypatch.setattr(
        run_module,
        "load_or_evaluate_initial",
        lambda *a, **k: initial_scores,
    )
    monkeypatch.setattr(
        run_module,
        "_initialize_optimization_state",
        lambda *a, **k: (decisions, initial_scores),
    )
    monkeypatch.setattr(
        run_module,
        "_finish_run_provenance",
        lambda *a, **k: None,
    )
    original_coordinator_run = coordinator.run

    def event_spy(system, initial, trial, policy, runtime, *, capacity=1):
        calls.append(
            {
                "system": system,
                "initial": initial,
                "trial": trial,
                "policy": policy,
                "runtime": runtime,
            }
        )
        return original_coordinator_run(
            system,
            initial,
            trial,
            policy,
            runtime,
            capacity=capacity,
        )

    monkeypatch.setattr(native_flow.coordinator, "run", event_spy)
    run_module._run_pipeline(OmegaConf.create({}))

    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call["trial"], native_flow.NativeFlow)
    assert call["initial"].system == call["system"].identity
    attempt_rounds = [
        attempt.key.named_round for attempt in call["runtime"].attempts()
    ]
    assert attempt_rounds == ["initial/0", "prepare/0"]


def test_native_flow_failure_marks_attempt_unknown_and_does_not_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, loss, stages, args = _shared_segment_state(tmp_path)
    initial_scores = _scores(tmp_path, "failure-initial", 0.25)
    decisions = SimpleNamespace(scores={0: 0.25}, last_accepted=0)
    calls = []
    observed = {}

    def initialize(args, loss_computer, initial_result, stage_manager, runtime_config):
        return decisions, initial_result

    def fail_prediction():
        calls.append("prediction")
        raise RuntimeError("synthetic model failure")

    def advance(
        args,
        loss_computer,
        current,
        stage_manager,
        runtime_config,
        decisions_arg,
        step,
    ):
        native_flow.native_call("prediction", fail_prediction)
        raise AssertionError("failed prediction returned")

    monkeypatch.setattr(run_module, "_initialize_optimization_state", initialize)
    monkeypatch.setattr(run_module, "_advance_optimization_state", advance)
    original_coordinator_run = coordinator.run

    def event_spy(system, initial, trial, policy, runtime, *, capacity=1):
        try:
            return original_coordinator_run(
                system,
                initial,
                trial,
                policy,
                runtime,
                capacity=capacity,
            )
        finally:
            observed["attempts"] = runtime.attempts()

    monkeypatch.setattr(native_flow.coordinator, "run", event_spy)
    with pytest.raises(OutcomeUnknown):
        run_module.run_optimization_loop(
            args,
            loss,
            initial_scores,
            stages,
            {},
        )

    assert calls == ["prediction"]
    attempts = observed["attempts"]
    failed = [item for item in attempts if item.key.named_round.startswith("prediction/")]
    assert len(failed) == 1
    assert failed[0].status == "unknown"

    execution = json.loads((tmp_path / "execution.json").read_text(encoding="utf-8"))
    assert execution["state"] == "failed"
    assert execution["advances"] == 0
    assert execution["attempts"] == [
        {
            "component": "prepare",
            "boundary": "",
            "status": "completed",
        },
        {
            "component": "prediction",
            "boundary": "native_prediction",
            "status": "unknown",
        },
    ]


def test_execution_summary_write_failure_does_not_mask_model_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, loss, stages, args = _shared_segment_state(tmp_path)
    initial_scores = _scores(tmp_path, "summary-failure-initial", 0.25)
    decisions = SimpleNamespace(scores={0: 0.25}, last_accepted=0)

    def initialize(args, loss_computer, initial_result, stage_manager, runtime_config):
        return decisions, initial_result

    def fail_prediction():
        raise RuntimeError("synthetic model failure")

    def advance(
        args,
        loss_computer,
        current,
        stage_manager,
        runtime_config,
        decisions_arg,
        step,
    ):
        native_flow.native_call("prediction", fail_prediction)

    def fail_summary(*args, **kwargs):
        raise OSError("synthetic diagnostic sink failure")

    monkeypatch.setattr(run_module, "_initialize_optimization_state", initialize)
    monkeypatch.setattr(run_module, "_advance_optimization_state", advance)
    monkeypatch.setattr(native_flow, "_write_execution_summary", fail_summary)

    with pytest.raises(OutcomeUnknown) as caught:
        run_module.run_optimization_loop(
            args,
            loss,
            initial_scores,
            stages,
            {},
        )

    messages = []
    error: BaseException | None = caught.value
    while error is not None:
        messages.append(str(error))
        error = error.__cause__ or error.__context__
    assert any("synthetic model failure" in message for message in messages)
    assert "synthetic diagnostic sink failure" not in str(caught.value)


def test_project_retains_exact_scores_and_live_pose_identity(
    tmp_path: Path,
) -> None:
    shared, _, stages, _ = _shared_segment_state(tmp_path)
    data = DataStore()
    translation = translate_legacy_state(assemblies=("AB",), data=data)
    resources = ProcessResources(data)
    pose = object()
    scores = StandardScores(
        names=["AB"],
        plddt={"AB": [np.asarray([0.4, 0.6, 0.4, 0.6], dtype=np.float32)]},
        complex_paths={"AB": tmp_path / "posed.cif"},
        complex_poses=ComplexPoses({"AB": pose}),
    )
    (tmp_path / "posed.cif").write_text("data_posed\n", encoding="utf-8")
    decisions = SimpleNamespace(scores={0: 0.4}, last_accepted=0)
    history = History(key_for(()), ())

    scored = native_flow._project(
        scores,
        translation,
        data,
        resources,
        decisions,
        stages,
        7,
        history,
    )

    assert scored.candidate == translation.candidate
    assert scored.assessment.values == (0.4,)
    resource_measurements = [
        measurement
        for measurement in scored.evidence.measurements
        if measurement.quantity == "legacy.process_resource"
    ]
    assert len(resource_measurements) == 1
    measurement = resource_measurements[0]
    restored = resources.resolve(
        measurement.values,
        candidate=translation.candidate.identity,
        revision=7,
    )
    assert restored is scores
    assert restored.complex_poses.poses["AB"] is pose
    assert shared is global_config.segments["shared"]
    resources.close()


def test_process_resource_codec_rejects_foreign_and_arbitrary_handles(
    tmp_path: Path,
) -> None:
    _, _, _, _ = _shared_segment_state(tmp_path)
    owner_data = DataStore()
    foreign_data = DataStore()
    owner = ProcessResources(owner_data)
    foreign = ProcessResources(foreign_data)
    try:
        foreign_blob = foreign.retain(object(), label="native-scores")
        with pytest.raises(ValueError, match="exact session codec"):
            owner.resolve(foreign_blob)

        with pytest.raises(ValueError, match="another session"):
            owner.codec.encode(
                ProcessReference(
                    session=foreign.nonce,
                    slot=0,
                    label="native-scores",
                )
            )
        with pytest.raises(TypeError, match="blob must be a Blob"):
            owner.resolve(object())  # type: ignore[arg-type]
    finally:
        foreign.close()
        owner.close()
