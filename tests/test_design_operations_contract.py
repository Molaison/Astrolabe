from dataclasses import dataclass
from functools import partial

import pytest
from jsonschema.exceptions import ValidationError

from astrolabe.design.operations import (
    Components, OperationOwner, _Prepared, _describe_operation, operation,
)
from astrolabe.design.types import key_for, manifest


@dataclass(frozen=True)
class Input:
    value: int


class FirstImplementation:
    pass


class SecondImplementation:
    pass


def test_selected_operation_is_validated_and_constructed_without_invocation(tmp_path):
    invoked = []

    class Owner(OperationOwner):
        pass

    def build(*, alias, options, context):
        return operation(
            alias, Input, Input,
            lambda value: invoked.append(value) or Input(value.value + options["offset"]),
            version="1", options=options, model_identity=key_for(options),
            schema={"type": "object", "properties": {"offset": {"type": "integer"}},
                    "required": ["offset"], "additionalProperties": False},
        )

    Owner.register_operation("shift", build)
    config = tmp_path / "selected.yaml"
    config.write_text("components:\n  change:\n    family: custom\n    type: shift\n    options: {offset: 2}\n")
    parts = Components.from_config(config, owners={"custom": Owner})
    assert invoked == []
    assert parts.selected_options() == {"change": {"offset": 2}}
    selected = parts.operation("change", Input, Input)
    assert selected.invoke(selected.prepare(Input(3))) == Input(5)
    assert invoked == [Input(3)]

    config.write_text("components:\n  change:\n    family: custom\n    type: shift\n    options: {offset: nope}\n")
    with pytest.raises(ValidationError):
        Components.from_config(config, owners={"custom": Owner})
    assert invoked == [Input(3)]


def test_identity_covers_implementation_and_accounting_boundary():
    first = _describe_operation("x", Input, Input, implementation=(FirstImplementation,), version="1")
    second = _describe_operation("x", Input, Input, implementation=(SecondImplementation,), version="1")
    assert first.identity != second.identity

    def identity(value):
        return value

    first = operation("x", Input, Input, identity, version="1", paid_boundary="prediction")
    second = operation("x", Input, Input, identity, version="1", paid_boundary="tool")
    assert first.descriptor.identity != second.descriptor.identity


def test_preparation_cannot_change_declared_accounting_boundary():
    component = operation(
        "x", Input, Input, lambda value: value, version="1",
        prepare=lambda request: _Prepared(request, manifest(()), (), "prediction"),
    )
    with pytest.raises(ValueError, match="accounting boundary"):
        component.prepare(Input(1))


def _shift(value, *, offset):
    return Input(value.value + offset)


def test_partial_requires_an_explicit_identity_for_its_captured_arguments():
    with pytest.raises(ValueError, match="stateful callable"):
        operation("shift", Input, Input, partial(_shift, offset=1), version="1")
    first = operation("shift", Input, Input, partial(_shift, offset=1), version="1", model_identity=key_for(1))
    second = operation("shift", Input, Input, partial(_shift, offset=9), version="1", model_identity=key_for(9))
    assert first.descriptor.identity != second.descriptor.identity


def test_default_argument_state_requires_an_explicit_identity():
    def make_shift(offset):
        def shift(value, offset=offset):
            return Input(value.value + offset)
        return shift

    def make_keyword_shift(offset):
        def shift(value, *, offset=offset):
            return Input(value.value + offset)
        return shift

    for factory in (make_shift, make_keyword_shift):
        with pytest.raises(ValueError, match="stateful callable"):
            operation("shift", Input, Input, factory(1), version="1")
        first = operation("shift", Input, Input, factory(1), version="1", model_identity=key_for(1))
        second = operation("shift", Input, Input, factory(9), version="1", model_identity=key_for(9))
        assert first.descriptor.identity != second.descriptor.identity


def test_prepared_request_is_detached_and_cannot_change_before_invocation():
    @dataclass
    class MutableInput:
        value: int

    def identity(value):
        return value

    selected = operation("x", MutableInput, MutableInput, identity, version="1")
    request = MutableInput(1)
    prepared = selected.prepare(request)
    request.value = 9
    assert selected.invoke(prepared).value == 1
    prepared.request.value = 8
    with pytest.raises(ValueError, match="prepared request changed"):
        selected.invoke(prepared)


def test_component_factory_receives_a_detached_nested_configuration():
    class Owner(OperationOwner):
        pass

    def build(*, alias, options, context):
        return operation(alias, Input, Input, lambda value: Input(value.value + options["nested"]["offset"]),
                         version="1", options=options, model_identity=key_for(options),
                         schema={"type": "object", "properties": {"nested": {"type": "object"}}})

    Owner.register_operation("shift", build)
    config = {"components": {"x": {"family": "custom", "type": "shift", "options": {"nested": {"offset": 1}}}}}
    parts = Components.from_config(config, owners={"custom": Owner})
    config["components"]["x"]["options"]["nested"]["offset"] = 9
    selected = parts.operation("x", Input, Input)
    assert parts.selected_options()["x"]["nested"]["offset"] == 1
    assert selected.invoke(selected.prepare(Input(1))) == Input(2)
