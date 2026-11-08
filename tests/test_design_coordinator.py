from dataclasses import replace
import random

import pytest

from astrolabe.design import (
    Assignment, DataStore, Evidence, JSONCodec, Polymer, Proposal, Runtime,
    Scored, TrialOutcome, key_for, make_candidate, make_system, manifest, operation,
    run,
)
from astrolabe.design.operations import Components
from astrolabe.design.search import BeamSearchPolicy
from astrolabe.design.types import Assessment, Candidate


def setup(data):
    topology = data.put({"bonds": []}, JSONCodec())
    system = make_system(("shared",), (), topology, ())
    candidate = make_candidate(system, (Assignment("shared", Polymer("protein", ("G", "G", "G"), ())),))
    return system, candidate


class CountingTrial:
    def __init__(self, system, calls):
        self.system, self.calls = system, calls
        def edit(candidate):
            calls["edit"] += 1
            value = candidate.assignments[0].value
            position = next((i for i, token in enumerate(value.tokens) if token == "G"), 2)
            tokens = list(value.tokens)
            tokens[position] = "A"
            return system.edit(candidate, (Assignment("shared", replace(value, tokens=tuple(tokens))),))
        def score(candidate):
            calls["score"] += 1
            inputs = manifest(())
            evidence = Evidence(key_for((candidate.identity, "evidence")), candidate.identity, (), (), inputs)
            assessment = Assessment(evidence.identity, key_for("objective"), key_for(()),
                (float(candidate.assignments[0].value.tokens.count("A")),), (), inputs)
            return Scored(candidate, evidence, assessment)
        self.edit = operation("edit", Candidate, Candidate, edit, version="1")
        self.score = operation("score", Candidate, Scored, score, version="1", paid_boundary="assessment")
        self.parts = Components({"edit": self.edit, "score": self.score})
        self.requirements = self.parts.requirements(("edit", "score"))
        self.identity = key_for(("counting-trial/v1", self.edit.descriptor.identity, self.score.descriptor.identity))

    def root(self, candidate, runtime):
        self.calls["root"] += 1
        return runtime.call("root/score", self.score, candidate)

    def perform(self, work, runtime):
        self.calls["perform"] += 1
        current = work.parents[0]
        proposals = []
        for round_index in range(2):
            base = current.candidate
            child = runtime.call(f"round/{round_index}/edit", self.edit, base)
            proposals.append(Proposal(key_for((work.identity, round_index, child.identity)),
                tuple(p.candidate.identity for p in work.parents), base.identity, child,
                (current.evidence.identity,), self.identity))
            current = runtime.call(f"round/{round_index}/score", self.score, child)
        return TrialOutcome(work.identity, "completed", tuple(proposals), (current,), ())


def make_policy():
    return BeamSearchPolicy(rounds=1, width=2, offspring=1, max_work=2,
                            seed=17, directions=("max",))


def test_root_two_round_trial_and_completed_resume_without_science_calls(tmp_path):
    data = DataStore(tmp_path / "data")
    system, initial = setup(data)
    calls = dict(edit=0, score=0, root=0, perform=0)
    trial = CountingTrial(system, calls)
    with Runtime(trial.parts, root=tmp_path, data=data) as runtime:
        first = run(system, initial, trial, make_policy(), runtime)
        assert first.selected[0].assessment.values == (2.0,)
        assert len(runtime.history.records) == 1
    before = dict(calls)
    with Runtime(trial.parts, root=tmp_path) as runtime:
        runtime.data.register(JSONCodec())
        resumed = run(system, initial, trial, make_policy(), runtime)
    assert resumed.selected == first.selected
    assert calls == before
    assert calls == dict(edit=2, score=3, root=1, perform=1)
    assert (tmp_path / "selected.json").is_file()


def test_result_before_tell_crash_recovers_outcome_without_repeating_trial(tmp_path, monkeypatch):
    data = DataStore(tmp_path / "data")
    system, initial = setup(data)
    calls = dict(edit=0, score=0, root=0, perform=0)
    trial = CountingTrial(system, calls)
    with Runtime(trial.parts, root=tmp_path, data=data) as runtime:
        original = runtime.save
        def interrupted(name, value, *args):
            if name == "run/state" and value["history"].records:
                raise KeyboardInterrupt()
            return original(name, value, *args)
        monkeypatch.setattr(runtime, "save", interrupted)
        with pytest.raises(KeyboardInterrupt):
            run(system, initial, trial, make_policy(), runtime)
    before = dict(calls)
    with Runtime(trial.parts, root=tmp_path) as runtime:
        runtime.data.register(JSONCodec())
        result = run(system, initial, trial, make_policy(), runtime)
        assert len(runtime.history.records) == 1
    assert result.selected[0].assessment.values == (2.0,)
    assert calls == before


def test_changed_trial_refuses_completed_resume(tmp_path):
    data = DataStore(tmp_path / "data")
    system, initial = setup(data)
    trial = CountingTrial(system, dict(edit=0, score=0, root=0, perform=0))
    with Runtime(trial.parts, root=tmp_path, data=data) as runtime:
        run(system, initial, trial, make_policy(), runtime)
    trial.identity = key_for("changed-trial-options")
    with Runtime(trial.parts, root=tmp_path) as runtime:
        with pytest.raises(ValueError, match="changed on resume"):
            run(system, initial, trial, make_policy(), runtime)


def test_saved_work_restores_exit_rng_before_a_new_sibling(tmp_path, monkeypatch):
    def execute(root, interrupt):
        data = DataStore(root / "data")
        system, initial = setup(data)
        calls = dict(edit=0, score=0, root=0, perform=0)
        trial = CountingTrial(system, calls)
        draws = []
        original_edit = trial.edit._invoke
        def draw_and_edit(candidate):
            draws.append(random.random())
            return original_edit(candidate)
        trial.edit = operation("edit", Candidate, Candidate, draw_and_edit, version="2")
        trial.parts = Components({"edit": trial.edit, "score": trial.score})
        trial.identity = key_for(("rng-trial", trial.edit.descriptor.identity))
        policy = BeamSearchPolicy(rounds=1, width=2, offspring=2, max_work=3,
                                  seed=17, directions=("max",))
        random.seed(71)
        with Runtime(trial.parts, root=root, data=data) as runtime:
            original_save = runtime.save
            crashed = False
            def crash_after_first_outcome(name, value, *args):
                nonlocal crashed
                result = original_save(name, value, *args)
                if interrupt and name.startswith("outcome/") and not crashed:
                    crashed = True
                    raise KeyboardInterrupt()
                return result
            monkeypatch.setattr(runtime, "save", crash_after_first_outcome)
            if interrupt:
                with pytest.raises(KeyboardInterrupt):
                    run(system, initial, trial, policy, runtime, capacity=2)
            else:
                run(system, initial, trial, policy, runtime, capacity=2)
        if interrupt:
            random.seed(981)
            with Runtime(trial.parts, root=root) as runtime:
                run(system, initial, trial, policy, runtime, capacity=2)
        return draws
    assert execute(tmp_path / "resumed", True) == execute(tmp_path / "continuous", False)


def test_composition_releases_its_owned_models_on_failure(tmp_path):
    from astrolabe.model_execution import register_model_owner
    closed = []
    class Owner:
        attached = False
        def shutdown(self):
            closed.append(self)
    owner = Owner()
    data = DataStore(tmp_path / "data")
    system, initial = setup(data)
    trial = CountingTrial(system, dict(edit=0, score=0, root=0, perform=0))
    def fail(request):
        register_model_owner(owner)
        raise ValueError("scientific operation failed")
    trial.score = operation("score", Candidate, Scored, fail, version="failure/v1")
    trial.parts = Components({"edit": trial.edit, "score": trial.score})
    with Runtime(trial.parts, root=tmp_path, data=data) as runtime:
        with pytest.raises(ValueError, match="scientific operation"):
            run(system, initial, trial, make_policy(), runtime)
    assert closed == [owner]
