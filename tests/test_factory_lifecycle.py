from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.evaluators.registry import EvaluatorRegistry
from astrolabe.evaluators.utils import construct_evaluators
from astrolabe.mutators import base as mutator_base
from astrolabe.mutators.registry import MutatorRegistry
from astrolabe.runtime_support import ResourceOwner
from astrolabe.evaluator_startup import EvaluatorInitializationError
from astrolabe.structure_processors.base import StructureProcessorRegistry, initialize_predockers


@pytest.fixture(autouse=True)
def reset_state():
    global_config.reset()
    yield
    global_config.reset()


class Resource:
    def __init__(self, name, events, failure=None):
        self.name = name
        self.events = events
        self.failure = failure

    def shutdown(self):
        self.events.append(self.name)
        if self.failure is not None:
            raise self.failure


def invoke(family, names, factory, tmp_path, monkeypatch, owner=None):
    terms = {name: {"_type": "fake"} for name in names}
    if family == "evaluator":
        monkeypatch.setattr(EvaluatorRegistry, "get", lambda name: factory)
        return construct_evaluators(OmegaConf.create({"evaluator_terms": terms}), tmp_path, resource_owner=owner)
    if family == "mutator":
        monkeypatch.setattr(MutatorRegistry, "get", lambda name: factory)
        monkeypatch.setattr(mutator_base, "validate_mutator_capabilities", lambda *a, **kw: None)
        return mutator_base.build_mutators(OmegaConf.create({"mutator_terms": terms}), tmp_path, resource_owner=owner)
    monkeypatch.setattr(StructureProcessorRegistry, "get", lambda name: factory)
    initialize_predockers(OmegaConf.create({"predocker_terms": terms}), resource_owner=owner)
    return global_config.predockers


@pytest.mark.parametrize("family", ["evaluator", "mutator", "predocker"])
def test_factory_failure_cleans_unique_objects_and_preserves_original(family, tmp_path, monkeypatch):
    events = []
    original = ValueError("factory failure")
    shared = Resource("shared", events, RuntimeError("cleanup failed"))
    interrupted_cleanup = Resource("interrupt", events, KeyboardInterrupt("cleanup interrupt"))
    last = Resource("last", events)
    objects = iter([shared, shared, interrupted_cleanup, last])
    def factory(*args, **kwargs):
        try:
            return next(objects)
        except StopIteration:
            raise original
    with pytest.raises(EvaluatorInitializationError if family == "evaluator" else ValueError) as error:
        invoke(family, ["a", "alias", "b", "c", "fail"], factory, tmp_path, monkeypatch)
    assert (error.value.__cause__ if family == "evaluator" else error.value) is original
    assert events == ["shared", "interrupt", "last"]
    if family == "predocker":
        assert global_config.predockers == {}


@pytest.mark.parametrize("family", ["evaluator", "mutator", "predocker"])
def test_success_transfers_objects_without_cleanup(family, tmp_path, monkeypatch):
    events = []
    shared = Resource("shared", events)
    result = invoke(family, ["a", "alias"], lambda *args, **kwargs: shared, tmp_path, monkeypatch)
    assert result["a"] is result["alias"] is shared
    assert events == []


@pytest.mark.parametrize("family", ["evaluator", "mutator", "predocker"])
def test_object_cleaned_when_alias_assignment_fails(family, tmp_path, monkeypatch):
    events = []
    class BadAlias(Resource):
        @property
        def alias(self):
            return None
        @alias.setter
        def alias(self, value):
            raise ValueError("alias failed")
    obj = BadAlias("resource", events)
    with pytest.raises(EvaluatorInitializationError if family == "evaluator" else ValueError, match="alias failed"):
        invoke(family, ["a"], lambda *args, **kwargs: obj, tmp_path, monkeypatch)
    assert events == ["resource"]


def test_mutator_validation_failure_also_cleans_just_constructed(tmp_path, monkeypatch):
    events = []
    obj = Resource("mutator", events)
    monkeypatch.setattr(MutatorRegistry, "get", lambda name: lambda *a, **kw: obj)
    def fail(*args, **kwargs):
        raise ValueError("validation failed")
    monkeypatch.setattr(mutator_base, "validate_mutator_capabilities", fail)
    with pytest.raises(ValueError, match="validation failed"):
        mutator_base.build_mutators(OmegaConf.create({"mutator_terms": {"a": {"_type": "fake"}}}), tmp_path)
    assert events == ["mutator"]


@pytest.mark.parametrize("failure_family", ["evaluator", "predocker"])
def test_shared_owner_deduplicates_across_families_and_later_run_cleanup(failure_family, tmp_path, monkeypatch):
    events = []
    shared = Resource("shared", events)
    owner = ResourceOwner()
    invoke("mutator", ["a"], lambda *a, **kw: shared, tmp_path, monkeypatch, owner)
    assert events == []
    counter = iter([shared])
    original = RuntimeError("later factory failed")
    def later(*args, **kwargs):
        try:
            return next(counter)
        except StopIteration:
            raise original
    with pytest.raises(RuntimeError) as error:
        invoke(failure_family, ["alias", "fail"], later, tmp_path, monkeypatch, owner)
    assert (error.value.__cause__ if failure_family == "evaluator" else error.value) is original
    assert events == ["shared"]
    owner.close()
    assert events == ["shared"]
