"""Mutation budgets count independent mutable sites, resolved at proposal time."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


@dataclass(frozen=True)
class MutationBudget:
    unit: Literal["count", "fraction"]
    value: int | float

    @classmethod
    def parse(cls, raw_value) -> "MutationBudget":
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, str)):
            raise TypeError("mutation_num must be a non-negative integer, percentage string, or 'all'")
        value = raw_value.strip() if isinstance(raw_value, str) else raw_value
        if value == "all":
            return cls("fraction", 1.0)
        if isinstance(value, int):
            if value < 0:
                raise ValueError("mutation_num must be finite and non-negative")
            return cls("count", value)
        percentage = isinstance(value, str) and value.endswith("%")
        try:
            number = float(value[:-1].strip() if percentage else value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid mutation_num: {raw_value!r}") from exc
        if not math.isfinite(number) or number < 0:
            raise ValueError("mutation_num must be finite and non-negative")
        if percentage:
            if number > 100:
                raise ValueError("mutation_num percentage must be between 0% and 100%")
            return cls("fraction", number / 100)
        if not number.is_integer():
            raise ValueError("mutation_num count must be an integer")
        return cls("count", int(number))

    def resolve(self, mutable_sites: int) -> int:
        if self.unit == "count":
            return int(self.value)
        if self.value == 0 or mutable_sites == 0:
            return 0
        return max(1, round(mutable_sites * self.value))


def count_mutable_sites(protomer_pool, *, segments=None) -> int:
    """Count shared Segment positions once and exclude protected occurrences."""
    eligible = set()
    protected = set()
    for protomer in protomer_pool.values():
        kind = getattr(protomer, "type", None)
        kind = getattr(kind, "value", kind)
        if kind is not None and kind not in {"p", "r", "d", "protein", "rna", "dna"}:
            continue
        mask = getattr(protomer, "mutable_mask", None)
        if mask is None:
            length = getattr(protomer, "mutable_length", None)
            if length is None:
                length = len(protomer.sequence)
            eligible.update((id(protomer), pos) for pos in range(length))
            continue
        ptms = getattr(protomer, "ptm_dict", {}) or {}
        slices = getattr(protomer, "components_slices", ())
        components = getattr(protomer, "components", ())
        identities = getattr(protomer, "component_ids", ())
        if not slices:
            slices = (slice(0, len(mask)),)
        for index, region in enumerate(slices):
            if segments is not None and components:
                identity = id(segments[components[index]])
            elif identities:
                identity = identities[index]
            else:
                identity = (id(protomer), index)
            for pos in range(region.start, region.stop):
                key = (identity, pos - region.start)
                if mask[pos] and pos not in ptms:
                    eligible.add(key)
                else:
                    protected.add(key)
    return len(eligible - protected)


def resolve_mutation_num(raw_value, *, protomer_pool=None, segments=None) -> int | None:
    """Resolve a stage spec using the current mutable graph, without sampling."""
    if raw_value is None:
        return None
    budget = MutationBudget.parse(raw_value)
    if budget.unit == "count":
        return budget.resolve(0)
    if protomer_pool is None:
        from astrolabe.core import global_config

        context = global_config.build_runtime_context()
        protomer_pool = context.mutable_protomers or context.protomers
        segments = context.get_const("segments", None)
    return budget.resolve(count_mutable_sites(protomer_pool, segments=segments))


def current_mutation_count(*, weight=1.0, maximum=None, runtime=None) -> int:
    """Read the active stage policy afresh, then apply its mutator allocation."""
    if runtime is None:
        from astrolabe.core import global_config

        runtime = global_config

    def read(name, default=None):
        try:
            return getattr(runtime, name)
        except (AttributeError, RuntimeError):
            return default

    stage_manager = read("stage_manager")
    raw_value = (
        stage_manager._resolve_stage_mutation_value()
        if stage_manager is not None
        else read("mutation_num", 0)
    )
    pool = read("mutable_protomers") or read("protomers", {})
    base_count = resolve_mutation_num(
        raw_value, protomer_pool=pool, segments=read("segments")
    )
    if base_count is None:
        base_count = 0
    if stage_manager is not None:
        runtime.set_variable("mutation_num", base_count)
    if isinstance(weight, bool) or not math.isfinite(float(weight)) or float(weight) < 0:
        raise ValueError("mutation weight must be finite and non-negative")
    count = round(base_count * float(weight))
    if maximum is not None:
        count = min(count, maximum)
    return count
