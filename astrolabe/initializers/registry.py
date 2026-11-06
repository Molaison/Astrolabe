"""Owning registry for initializer implementations.

The registry stores import targets and callback requirements only. Constructing
an initializer does not import a model or execute scientific initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import importlib
from typing import Any, Callable, Mapping

from astrolabe.design.operations import OperationOwner


InitializerCallback = Callable[..., None]


@dataclass(frozen=True)
class InitializerRegistration:
    name: str
    module: str
    class_name: str
    callback_key: str | None = None
    fixed_residue_set: tuple[str, ...] | None = None
    requires_mutators: bool = False
    uses_segment_plan: bool = False


class InitializerRegistry(OperationOwner):
    """Registry for initializer classes and their callback construction."""

    _registrations: dict[str, InitializerRegistration] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        module: str,
        class_name: str,
        callback_key: str | None = None,
        fixed_residue_set: tuple[str, ...] | None = None,
        requires_mutators: bool = False,
        uses_segment_plan: bool = False,
    ) -> InitializerRegistration:
        if not name or not module or not class_name:
            raise ValueError("initializer name, module, and class_name are required")
        registration = InitializerRegistration(
            name=name,
            module=module,
            class_name=class_name,
            callback_key=callback_key,
            fixed_residue_set=(
                None
                if fixed_residue_set is None
                else tuple(str(residue) for residue in fixed_residue_set)
            ),
            requires_mutators=bool(requires_mutators),
            uses_segment_plan=bool(uses_segment_plan),
        )
        cls._registrations[name] = registration
        return registration

    @classmethod
    def get(cls, name: str) -> InitializerRegistration:
        try:
            return cls._registrations[name]
        except KeyError as exc:
            available = ", ".join(cls.list_available()) or "<none>"
            raise ValueError(
                f"Unknown initializer.mode={name!r}. Expected one of: {available}."
            ) from exc

    @classmethod
    def list_available(cls) -> tuple[str, ...]:
        return tuple(cls._registrations)

    @classmethod
    def construct(
        cls,
        name: str,
        *,
        args: Any,
        mutators: Any,
        segment_plan: Mapping[str, Any],
        callbacks: Mapping[str, InitializerCallback],
    ) -> Any:
        registration = cls.get(name)
        if registration.requires_mutators and mutators is None:
            raise ValueError(
                f"initializer.mode={name!r} requires initialized mutators."
            )

        module = importlib.import_module(registration.module)
        initializer_cls = getattr(module, registration.class_name)
        if registration.callback_key is not None:
            try:
                callback = callbacks[registration.callback_key]
            except KeyError as exc:
                raise ValueError(
                    f"initializer.mode={name!r} requires callback "
                    f"{registration.callback_key!r}"
                ) from exc
            if not callable(callback):
                raise TypeError(
                    f"initializer callback {registration.callback_key!r} is not callable"
                )
            apply_fn = partial(callback, args)
            if registration.fixed_residue_set is not None:
                apply_fn = partial(
                    callback,
                    args,
                    fixed_residue_set=list(registration.fixed_residue_set),
                )
            return initializer_cls(apply_fn)

        if registration.uses_segment_plan:
            return initializer_cls(
                args=args,
                mutators=mutators,
                segment_plan=segment_plan,
            )
        return initializer_cls()


def _register_defaults() -> None:
    defaults = (
        ("random", "astrolabe.initializers.builtin", "RandomInitializer", None, None, False, False),
        ("dplm", "astrolabe.initializers.builtin", "DPLMInitializer", "dplm", None, False, False),
        ("aa_set", "astrolabe.initializers.builtin", "AminoAcidSetInitializer", "aa_set", None, False, False),
        ("mask_mix", "astrolabe.initializers.builtin", "MaskMixInitializer", "mask_mix", None, False, False),
        ("all_a", "astrolabe.initializers.builtin", "AllAInitializer", "aa_set", ("A",), False, False),
        ("pssm", "astrolabe.initializers.builtin", "PSSMInitializer", "pssm", None, False, False),
        ("x_mix", "astrolabe.initializers.builtin", "XMixInitializer", "x_mix", None, False, False),
        ("structure", "astrolabe.initializers.structure", "ExistingStructureInitializer", None, None, True, True),
        ("rfd3", "astrolabe.initializers.rfd3", "RFD3Initializer", None, None, True, True),
    )
    for name, module, class_name, callback_key, fixed, requires_mutators, uses_segment_plan in defaults:
        InitializerRegistry.register(
            name,
            module=module,
            class_name=class_name,
            callback_key=callback_key,
            fixed_residue_set=fixed,
            requires_mutators=requires_mutators,
            uses_segment_plan=uses_segment_plan,
        )


_register_defaults()


def construct_initializer(
    mode: str,
    *,
    args: Any,
    mutators: Any,
    segment_plan: Mapping[str, Any],
    callbacks: Mapping[str, InitializerCallback],
) -> Any:
    return InitializerRegistry.construct(
        mode,
        args=args,
        mutators=mutators,
        segment_plan=segment_plan,
        callbacks=callbacks,
    )


__all__ = [
    "InitializerCallback",
    "InitializerRegistration",
    "InitializerRegistry",
    "construct_initializer",
]
