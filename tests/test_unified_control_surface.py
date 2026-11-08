from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pytest

from astrolabe.app import Astrolabe
from astrolabe.design import coordinator
from astrolabe.design.collection import CollectionConfig
from astrolabe.design.collection_search import (
    CollectionPolicy,
    CollectionRun,
    CollectionTrial,
    run_collection_search,
)
from astrolabe.design.composition import ControlRun
from astrolabe.design.types import key_for
from astrolabe.model_execution import capture_random_state


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _CounterState:
    completed: int = 0


@dataclass(frozen=True)
class _CounterRequest:
    pass


@dataclass(frozen=True)
class _CounterEvent:
    completed: int


@dataclass(frozen=True)
class _CounterSelection:
    completed: int


class _CounterPolicy:
    request_type = _CounterRequest
    result_type = _CounterEvent

    def ask(self, state: _CounterState) -> _CounterRequest:
        if self.finished(state):
            raise ValueError("finished policy cannot ask")
        return _CounterRequest()

    def tell(
        self,
        state: _CounterState,
        event: _CounterEvent,
    ) -> _CounterState:
        if event.completed != state.completed + 1:
            raise ValueError("control event is out of order")
        return _CounterState(event.completed)

    def finished(self, state: _CounterState) -> bool:
        return state.completed >= 1

    def selected(self, state: _CounterState) -> _CounterSelection:
        return _CounterSelection(state.completed)


class _CounterTrial:
    def __init__(self) -> None:
        self.requests: list[_CounterRequest] = []

    def perform(self, request: _CounterRequest) -> _CounterEvent:
        self.requests.append(request)
        return _CounterEvent(1)


class _RngDrawingPolicy(_CounterPolicy):
    def ask(self, state: _CounterState) -> _CounterRequest:
        random.random()
        np.random.random()
        return super().ask(state)


def _module(path: str) -> ast.Module:
    return ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)


def _function(
    path: str,
    name: str,
    *,
    owner: str | None = None,
) -> ast.FunctionDef:
    module = _module(path)
    scope: ast.AST = module
    if owner is not None:
        owners = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.ClassDef) and node.name == owner
        ]
        assert len(owners) == 1, f"{path}: expected one class {owner}"
        scope = owners[0]
    functions = [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(functions) == 1, f"{path}: expected one function {name}"
    return functions[0]


def _identifiers(node: ast.AST) -> set[str]:
    values: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            values.add(child.id)
        elif isinstance(child, ast.Attribute):
            values.add(child.attr)
        elif isinstance(child, ast.alias):
            values.add(child.name.rsplit(".", 1)[-1])
            if child.asname is not None:
                values.add(child.asname)
    return values


def _calls(node: ast.AST) -> set[str]:
    return {
        ast.unparse(child.func)
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
    }


def test_astrolabe_controlrun_uses_the_shared_drive_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    drive = coordinator.drive

    def _spy(policy, trial, state, *, transition=None):
        calls.append((policy, trial, state, transition))
        return drive(policy, trial, state, transition=transition)

    monkeypatch.setattr(coordinator, "drive", _spy)
    policy = _CounterPolicy()
    trial = _CounterTrial()
    initial = _CounterState()

    result = Astrolabe(
        ControlRun(
            policy=policy,
            trial=trial,
            initial_state=initial,
            runtime=None,
        )
    ).run()

    assert result == _CounterSelection(1)
    assert len(trial.requests) == 1
    assert calls == [(policy, trial, initial, None)]


def test_run_collection_search_uses_public_astrolabe_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    import astrolabe.model_execution as model_execution

    calls = []
    scopes = []
    drive = coordinator.drive

    def _spy(policy, trial, state, *, transition=None):
        calls.append((policy, trial, state, transition))
        return drive(policy, trial, state, transition=transition)

    @contextmanager
    def _scope():
        scopes.append("entered")
        yield

    monkeypatch.setattr(coordinator, "drive", _spy)
    monkeypatch.setattr(model_execution, "model_run_scope", _scope)
    policy = CollectionPolicy(
        config=CollectionConfig(
            num_jobs=1,
            min_completed=1,
            max_wait_s=1.0,
        ),
        decision_rounds=1,
        initial_payload=None,
    )
    with CollectionTrial(
        config=policy.config,
        worker=lambda _job: 0.5,
        selector=policy.selector,
    ) as trial:
        result = run_collection_search(policy, trial)

    assert isinstance(result, CollectionRun)
    assert scopes == ["entered"]
    assert len(calls) == 1
    assert calls[0][0] is policy
    assert calls[0][1] is trial
    assert calls[0][3] is None


def test_process_lifetime_policy_rng_draw_is_rejected_and_restored() -> None:
    random.seed(17)
    np.random.seed(23)
    before = capture_random_state()
    trial = _CounterTrial()

    with pytest.raises(
        ValueError,
        match="Control policy transitions must be pure",
    ):
        Astrolabe(
            ControlRun(
                policy=_RngDrawingPolicy(),
                trial=trial,
                initial_state=_CounterState(),
                runtime=None,
            )
        ).run()

    assert key_for(capture_random_state()) == key_for(before)
    assert trial.requests == []


def test_migrated_control_entries_forward_without_private_search_loops() -> None:
    entries = {
        ("run.py", "run_optimization_loop", None): {"run_native_optimization"},
        (
            "astrolabe/design/native_flow.py",
            "run_native_optimization",
            None,
        ): {"coordinator"},
        (
            "astrolabe/design/native_refinement.py",
            "run_native_refinement",
            None,
        ): {"ControlRun"},
        (
            "astrolabe/experimental/candidate_batch_runner.py",
            "run",
            "CandidateDecisionRun",
        ): {"Astrolabe", "ControlRun"},
        (
            "astrolabe/protein_beam.py",
            "run_beam",
            None,
        ): {"Astrolabe", "ControlRun"},
    }

    for (path, name, owner), expected in entries.items():
        function = _function(path, name, owner=owner)
        identifiers = _identifiers(function)
        assert expected <= identifiers, (path, name, expected - identifiers)
        assert not any(
            isinstance(node, ast.While) for node in ast.walk(function)
        ), f"{path}:{name} owns a while control loop"

    thin_forwarders = [
        ("run.py", "run_optimization_loop", None),
        (
            "astrolabe/design/native_refinement.py",
            "run_native_refinement",
            None,
        ),
        (
            "astrolabe/experimental/candidate_batch_runner.py",
            "run",
            "CandidateDecisionRun",
        ),
    ]
    for path, name, owner in thin_forwarders:
        function = _function(path, name, owner=owner)
        assert not any(
            isinstance(node, (ast.For, ast.AsyncFor))
            for node in ast.walk(function)
        ), f"{path}:{name} owns a generation loop"


def test_rna_public_facades_forward_without_a_private_control_loop() -> None:
    module = _module("astrolabe/rna_beam.py")
    public_names = (
        "run_rna_beam",
        "run_bivalent_rna_beam",
        "run_configured_refinement",
    )
    private_control_names = {
        node.name
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_run_native_rna_beam" not in private_control_names
    assert not any(isinstance(node, ast.While) for node in ast.walk(module))

    for name in public_names:
        function = _function("astrolabe/rna_beam.py", name)
        assert not any(
            isinstance(node, ast.While) for node in ast.walk(function)
        ), f"rna_beam.py:{name} owns a while control loop"
        assert any(
            isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
            for node in ast.walk(function)
        ), f"rna_beam.py:{name} does not forward to its bounded adapter"
    assert "run_native_refinement" in _identifiers(module)


def test_paper_run_does_not_dispatch_by_engine_or_rna_protocol() -> None:
    function = _function("astrolabe/paper.py", "_run")
    identifiers = _identifiers(function)
    literals = {
        node.value.casefold()
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "run_count" not in identifiers
    assert not any("rna" in value for value in literals)
    assert not any(
        isinstance(node, ast.While) for node in ast.walk(function)
    )
    assert not any(
        isinstance(node, (ast.For, ast.AsyncFor, ast.While))
        and "engine" in _identifiers(node)
        for node in ast.walk(function)
    )
    assert not any(
        isinstance(node, ast.For) and "run_count" in _identifiers(node)
        for node in ast.walk(function)
    )
    assert "cli._run_from_config" in _calls(function)


def test_retired_budget_state_generation_fill_and_dispatch_are_absent() -> None:
    sources = [ROOT / "run.py", *(ROOT / "astrolabe").rglob("*.py")]
    retired = {
        "BeamSearchState",
        "fill_generation",
        "protein_beam_dispatch",
    }
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert retired.isdisjoint(_identifiers(tree)), path.relative_to(ROOT)

    assert not (ROOT / "astrolabe/protein_beam_dispatch.py").exists()
