from dataclasses import dataclass

import pytest

from astrolabe.design.operations import (
    Components,
    OperationOwner,
    _Prepared,
    _describe_operation,
    operation,
)
from astrolabe.design.runtime import Runtime
from astrolabe.design.types import Dependency, History, key_for, manifest


@dataclass(frozen=True)
class Input:
    value: int


class FirstImplementation:
    pass


class SecondImplementation:
    pass


def test_new_family_is_owned_and_cold_config_shows_only_selected_options(tmp_path):
    invoked = []
    class Owner(OperationOwner):
        pass
    def build(*, alias, options, context):
        return operation(alias, Input, Input,
            lambda value: invoked.append(value) or Input(value.value + options["offset"]),
            version="1", options=options,
            schema={"type": "object", "properties": {"offset": {"type": "integer"}},
                    "required": ["offset"], "additionalProperties": False})
    Owner.register_operation("shift", build)
    config = tmp_path / "selected.yaml"
    config.write_text("components:\n  change:\n    family: custom\n    type: shift\n    options: {offset: 2}\n")
    parts = Components.from_config(config, owners={"custom": Owner})
    assert invoked == []
    assert parts.selected_options() == {"change": {"offset": 2}}
    component = parts.operation("change", Input, Input)
    with Runtime(parts) as runtime:
        runtime.preflight(parts.requirements(("change",)))
        with runtime.bind("work", History(key_for(()), ())):
            assert runtime.call("change", component, Input(3)) == Input(5)
    assert invoked == [Input(3)]


def test_preparation_cannot_draw_rng_before_issue():
    import random
    def prepare(request):
        random.random()
        return _Prepared(request, manifest((Dependency("input", key_for(request)),)), (), "")
    component = operation("x", Input, Input, lambda x: x, version="1", prepare=prepare)
    parts = Components({"x": component})
    with Runtime(parts) as runtime:
        runtime.preflight(parts.requirements(("x",)))
        with runtime.bind("work", History(key_for(()), ())):
            before = random.getstate()
            with pytest.raises(ValueError, match="preparation consumed RNG"):
                runtime.call("prepare", component, Input(1))
            assert random.getstate() == before
            assert runtime.attempts() == ()


def test_accounting_boundary_is_part_of_implementation_identity():
    def identity(value):
        return value
    first = operation("x", Input, Input, identity, version="1", paid_boundary="prediction")
    second = operation("x", Input, Input, identity, version="1", paid_boundary="tool")
    assert first.descriptor.identity != second.descriptor.identity


def test_descriptor_identity_hashes_the_actual_implementation_classes():
    first = _describe_operation(
        "x",
        Input,
        Input,
        implementation=(FirstImplementation,),
        version="1",
    )
    second = _describe_operation(
        "x",
        Input,
        Input,
        implementation=(SecondImplementation,),
        version="1",
    )
    assert first.identity != second.identity


def test_descriptor_identity_includes_lifecycle_callbacks():
    def prepare_one(request):
        return request

    def prepare_two(request):
        return request

    def result_one(result):
        return result

    def result_two(result):
        return result

    base = dict(
        alias="x",
        request_type=Input,
        result_type=Input,
        implementation=(FirstImplementation,),
        version="1",
    )
    first = _describe_operation(**base, prepare=prepare_one, on_result=result_one)
    second = _describe_operation(**base, prepare=prepare_two, on_result=result_two)
    assert first.identity != second.identity


def test_preparation_cannot_change_its_declared_accounting_boundary():
    component = operation("x", Input, Input, lambda x: x, version="1",
        prepare=lambda request: _Prepared(request, manifest(()), (), "prediction"))
    with pytest.raises(ValueError, match="accounting boundary"):
        component.prepare(Input(1))
