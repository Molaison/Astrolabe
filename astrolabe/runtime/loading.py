"""Model loading boundaries; resource admission belongs to the kernel handler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar


Prepared = TypeVar("Prepared")


@dataclass(frozen=True)
class StagedLoader(Generic[Prepared]):
    """Prepare on CPU, then place under the kernel's device/RNG scope.

    ``prepare(settings)`` returns opaque prepared state without placing a model
    on its target device. ``place(prepared)`` returns the inference callable (or
    KernelStages) and metadata. The handler owns leases and device completion;
    neither callback changes the scientific settings or chooses a new device.
    """

    prepare: Callable[[dict[str, Any]], Prepared]
    place: Callable[[Prepared], tuple[Any, dict[str, Any]]]
