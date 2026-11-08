"""Typed projection fixtures for legacy stage unit tests.

The legacy tests in this module family exercise decisions and stage
transactions with artificial evaluation objects rather than molecular graphs.
These projections replace only graph conversion and score projection; the
coordinator, Runtime, named native phases, decisions and stage owners remain
real.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from astrolabe.design import native_flow
from astrolabe.design.types import (
    Assessment,
    Assignment,
    Blob,
    Dependency,
    Domain,
    Evidence,
    Polymer,
    Scored,
    key_for,
    make_candidate,
    make_system,
    manifest,
)


_SYSTEM = make_system(
    variables=("test_variable",),
    occurrences=(),
    topology=Blob(
        key_for(("native-flow-test-topology/v1",)),
        "native-flow-test-topology/v1",
    ),
    domains=(Domain("test_variable", (0,), (), None),),
)
_INITIAL = make_candidate(
    _SYSTEM,
    (
        Assignment(
            "test_variable",
            Polymer("protein", ("G",), ()),
        ),
    ),
)
_TRANSLATION = SimpleNamespace(
    system=_SYSTEM,
    candidate=_INITIAL,
    state="test",
)


def _translate(_stage_manager, _data):
    return _TRANSLATION


def _score_value(decisions) -> float:
    scores = getattr(decisions, "scores", None)
    last_accepted = getattr(decisions, "last_accepted", None)
    if scores is None or last_accepted is None:
        return 0.0
    try:
        return float(scores[last_accepted])
    except (IndexError, KeyError, TypeError, ValueError):
        return 0.0


def _project(
    _scores,
    translation,
    _data,
    _resources,
    decisions,
    stage_manager,
    revision,
    history,
):
    candidate = translation.candidate
    stage = getattr(stage_manager, "current_stage", "default") or "default"
    objective = key_for(("native-flow-test-stage-objective/v1", stage))
    inputs = manifest(
        (
            Dependency("objective", objective),
            Dependency("history", history.revision),
        )
    )
    evidence = Evidence(
        identity=key_for(
            (
                "native-flow-test-evidence/v1",
                candidate.identity,
                revision,
                history.revision,
            )
        ),
        evaluated_candidate=candidate.identity,
        artifacts=(),
        measurements=(),
        inputs=inputs,
    )
    return Scored(
        candidate,
        evidence,
        Assessment(
            evidence=evidence.identity,
            objective_revision=objective,
            history_revision=history.revision,
            values=(_score_value(decisions),),
            gates=(),
            inputs=inputs,
        ),
    )


@pytest.fixture
def native_flow_test_projections(monkeypatch):
    monkeypatch.setattr(native_flow, "_translate", _translate)
    monkeypatch.setattr(native_flow, "_project", _project)
    yield _TRANSLATION


__all__ = ["native_flow_test_projections"]
