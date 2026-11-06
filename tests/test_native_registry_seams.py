from __future__ import annotations

from types import SimpleNamespace

import pytest

import astrolabe.initializers.registry as initializer_registry
import astrolabe.selectors.registry as selector_registry
from astrolabe.design.operations import OperationOwner
from astrolabe.initializers.registry import InitializerRegistry
from astrolabe.losses.utils import LossRegistry
from astrolabe.per_residue_scores import PerResidueScoreRegistry
from astrolabe.selectors.registry import SelectorRegistry


def test_existing_score_and_loss_registries_expose_operation_owner() -> None:
    assert issubclass(PerResidueScoreRegistry, OperationOwner)
    assert issubclass(LossRegistry, OperationOwner)

    def score_factory(*, alias, options, context):
        return ("score-operation", alias, options, context)

    def loss_factory(*, alias, options, context):
        return ("loss-operation", alias, options, context)

    PerResidueScoreRegistry.register_operation("unit_score", score_factory)
    loss_registry = LossRegistry()
    loss_registry.register_operation("unit_loss", loss_factory)

    assert PerResidueScoreRegistry.build_operation(
        "unit_score",
        alias="score",
        options={},
        context=None,
    ) == ("score-operation", "score", {}, None)
    assert loss_registry.build_operation(
        "unit_loss",
        alias="loss",
        options={},
        context=None,
    ) == ("loss-operation", "loss", {}, None)


def test_selector_registry_preserves_names_contexts_and_pose_metadata() -> None:
    assert set(SelectorRegistry.list_available()) == {
        "sse",
        "cdr",
        "bfactor",
        "interface",
        "surface",
        "surface_high_sap",
        "buried_unsat",
        "cavity_neighbor",
        "ligand_unsat",
        "ligand_contact",
        "membrane_outer_surface",
        "hotspot",
        "hole_surface",
        "pore_bottleneck",
        "long_helix_core",
        "long_loop",
        "rosetta_interface_energy",
    }
    membrane = SelectorRegistry.get("membrane_outer_surface")
    assert membrane.context == "complex_poses"
    assert membrane.pose_required is True
    assert membrane.path_only is True

    rosetta = SelectorRegistry.get("rosetta_interface_energy")
    assert rosetta.context == "evaluation_result,protomers"
    assert rosetta.pose_required is False

    hotspot = SelectorRegistry.get("hotspot")
    assert hotspot.pose_required is False


def test_selector_environment_imports_targets_lazily_and_selects_context(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class CapturingSelector:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def fake_import(module: str):
        calls.append(module)
        if module == "astrolabe.selectors.interface":
            return SimpleNamespace(InterfacePositionSelector=CapturingSelector)
        if module == "astrolabe.selectors.rosetta_interface_energy":
            return SimpleNamespace(
                RosettaInterfaceEnergyPositionSelector=CapturingSelector
            )
        raise AssertionError(module)

    monkeypatch.setattr(
        selector_registry.importlib,
        "import_module",
        fake_import,
    )
    poses = object()
    evaluation = object()
    protomers = object()

    environment = SelectorRegistry.construct_environment(
        {"interface", "rosetta_interface_energy"},
        complex_poses=poses,
        evaluation_result=evaluation,
        protomers=protomers,
    )

    assert set(calls) == {
        "astrolabe.selectors.interface",
        "astrolabe.selectors.rosetta_interface_energy",
    }
    interface = environment["interface"]()
    assert interface.args == (poses,)
    assert interface.kwargs == {"merge": "any"}

    rosetta = environment["rosetta_interface_energy"]()
    assert rosetta.args == (evaluation, protomers)
    assert rosetta.kwargs == {}


def test_initializer_registry_preserves_canonical_modes() -> None:
    assert set(InitializerRegistry.list_available()) == {
        "random",
        "dplm",
        "aa_set",
        "mask_mix",
        "all_a",
        "pssm",
        "x_mix",
        "structure",
        "rfd3",
    }
    assert InitializerRegistry.get("structure").requires_mutators is True
    assert InitializerRegistry.get("structure").uses_segment_plan is True
    assert InitializerRegistry.get("rfd3").requires_mutators is True
    assert InitializerRegistry.get("rfd3").uses_segment_plan is True


def test_all_a_initializer_callback_is_deferred_until_apply() -> None:
    calls = []
    args = SimpleNamespace(task_name="callback-test")

    def callback(*values, **kwargs):
        calls.append((values, kwargs))

    initializer = InitializerRegistry.construct(
        "all_a",
        args=args,
        mutators=None,
        segment_plan={},
        callbacks={"aa_set": callback},
    )
    assert calls == []

    result = initializer.apply()
    assert result.mode == "all_a"
    assert calls == [((args,), {"fixed_residue_set": ["A"]})]


def test_structure_initializer_receives_mutators_and_segment_plan(monkeypatch) -> None:
    captured = {}

    class FakeStructureInitializer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        initializer_registry.importlib,
        "import_module",
        lambda module: SimpleNamespace(
            ExistingStructureInitializer=FakeStructureInitializer
        ),
    )
    args = object()
    mutators = {"ligandmpnn": object()}
    segment_plan = {"A": [("A_1", slice(0, 3), [0, 1, 2])]}

    with pytest.raises(ValueError, match="requires initialized mutators"):
        InitializerRegistry.construct(
            "structure",
            args=args,
            mutators=None,
            segment_plan=segment_plan,
            callbacks={},
        )

    initializer = InitializerRegistry.construct(
        "structure",
        args=args,
        mutators=mutators,
        segment_plan=segment_plan,
        callbacks={},
    )
    assert isinstance(initializer, FakeStructureInitializer)
    assert captured == {
        "args": args,
        "mutators": mutators,
        "segment_plan": segment_plan,
    }


def test_unknown_initializer_mode_uses_registry_name_list() -> None:
    with pytest.raises(ValueError, match="Expected one of: random, dplm"):
        InitializerRegistry.get("not-an-initializer")
