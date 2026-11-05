"""Typed selector ownership and expression-environment construction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import importlib
from typing import Any

from astrolabe.design.operations import OperationOwner


@dataclass(frozen=True)
class SelectorRegistration:
    name: str
    module: str
    class_name: str
    context: str
    pose_required: bool
    path_only: bool = False


class SelectorRegistry(OperationOwner):
    """The owning registry for position selectors.

    Context is one of:
    - ``complex_poses``: construct with ``(complex_poses, merge="any")``
    - ``evaluation_result,protomers``: construct with those two positional values
    """

    _registrations: dict[str, SelectorRegistration] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        module: str,
        class_name: str,
        context: str,
        pose_required: bool,
        path_only: bool = False,
    ) -> SelectorRegistration:
        if not name or not module or not class_name:
            raise ValueError("selector name, module, and class_name are required")
        if context not in {"complex_poses", "evaluation_result,protomers"}:
            raise ValueError(f"unsupported selector context {context!r}")
        registration = SelectorRegistration(
            name=name,
            module=module,
            class_name=class_name,
            context=context,
            pose_required=bool(pose_required),
            path_only=bool(path_only),
        )
        cls._registrations[name] = registration
        return registration

    @classmethod
    def get(cls, name: str) -> SelectorRegistration:
        try:
            return cls._registrations[name]
        except KeyError as exc:
            raise ValueError(f"Invalid variable: {name}") from exc

    @classmethod
    def list_available(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registrations))

    @classmethod
    def pose_required_variables(cls) -> frozenset[str]:
        return frozenset(
            name
            for name, registration in cls._registrations.items()
            if registration.pose_required
        )

    @classmethod
    def path_only_variables(cls) -> frozenset[str]:
        return frozenset(
            name
            for name, registration in cls._registrations.items()
            if registration.path_only
        )

    @classmethod
    def construct_environment(
        cls,
        variables: set[str],
        *,
        complex_poses: Any,
        evaluation_result: Any,
        protomers: Any,
    ) -> dict[str, Any]:
        environment: dict[str, Any] = {}
        for variable in variables:
            registration = cls.get(variable)
            module = importlib.import_module(registration.module)
            selector_cls = getattr(module, registration.class_name)
            if registration.context == "complex_poses":
                environment[variable] = partial(
                    selector_cls,
                    complex_poses,
                    merge="any",
                )
            else:
                environment[variable] = partial(
                    selector_cls,
                    evaluation_result,
                    protomers,
                )
        return environment


def _register_defaults() -> None:
    defaults = (
        ("sse", "astrolabe.selectors.sse", "SecondaryStructurePositionSelector", "complex_poses", True, False),
        ("cdr", "astrolabe.selectors.cdr", "CDRPositionSelector", "complex_poses", True, False),
        ("bfactor", "astrolabe.selectors.bfactor", "BFactorPositionSelector", "complex_poses", True, False),
        ("interface", "astrolabe.selectors.interface", "InterfacePositionSelector", "complex_poses", True, False),
        ("surface", "astrolabe.selectors.surface", "SurfacePositionSelector", "complex_poses", True, False),
        ("surface_high_sap", "astrolabe.selectors.protein_quality", "SurfaceHighSAPPositionSelector", "complex_poses", True, False),
        ("buried_unsat", "astrolabe.selectors.protein_quality", "BuriedUnsatPositionSelector", "complex_poses", True, False),
        ("cavity_neighbor", "astrolabe.selectors.protein_quality", "CavityNeighborPositionSelector", "complex_poses", True, False),
        ("ligand_unsat", "astrolabe.selectors.ligand", "LigandUnsatisfiedPolarPositionSelector", "complex_poses", True, False),
        ("ligand_contact", "astrolabe.selectors.ligand", "LigandContactPositionSelector", "complex_poses", True, False),
        ("membrane_outer_surface", "astrolabe.selectors.ligand", "MembraneOuterSurfacePositionSelector", "complex_poses", True, True),
        ("hotspot", "astrolabe.selectors.hotspot", "HotspotAliasPositionSelector", "complex_poses", False, False),
        ("hole_surface", "astrolabe.selectors.hole", "HoleSurfacePositionSelector", "complex_poses", False, False),
        ("pore_bottleneck", "astrolabe.selectors.hole", "PoreBottleneckPositionSelector", "complex_poses", False, False),
        ("long_helix_core", "astrolabe.selectors.sse", "LongHelixCorePositionSelector", "complex_poses", True, False),
        ("long_loop", "astrolabe.selectors.sse", "LongLoopPositionSelector", "complex_poses", True, False),
        ("rosetta_interface_energy", "astrolabe.selectors.rosetta_interface_energy", "RosettaInterfaceEnergyPositionSelector", "evaluation_result,protomers", False, False),
    )
    for name, module, class_name, context, pose_required, path_only in defaults:
        SelectorRegistry.register(
            name,
            module=module,
            class_name=class_name,
            context=context,
            pose_required=pose_required,
            path_only=path_only,
        )


_register_defaults()


__all__ = ["SelectorRegistration", "SelectorRegistry"]
