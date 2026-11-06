"""Import-cheap stage configuration semantics shared with preflight."""

from __future__ import annotations

from typing import Optional


def resolve_mutation_num(raw_value, *, protomer_pool=None) -> Optional[int]:
    """Convert a mutation_num spec into an integer count."""

    if raw_value is None:
        return None

    def _total_mutable_length():
        pool = protomer_pool
        if pool is None:
            from astrolabe.core import global_config

            pool = (
                global_config.mutable_protomers
                if getattr(global_config, "mutable_protomers", None)
                else global_config.protomers
            )
        total = 0
        for protomer in pool.values():
            try:
                total += protomer.mutable_length
            except AttributeError:
                total += len(protomer.sequence)
        return total or 1

    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped.endswith("%"):
            number_part = stripped[:-1].strip()
            if not number_part:
                raise ValueError("Percentage value for mutation_num is empty.")
            try:
                percent = float(number_part) / 100.0
            except ValueError as exc:
                raise ValueError(
                    f"Invalid percentage value for mutation_num: '{raw_value}'"
                ) from exc
            total_len = _total_mutable_length()
            return max(1, int(round(total_len * percent)))
        try:
            return int(round(float(stripped)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric value for mutation_num: '{raw_value}'"
            ) from exc

    if isinstance(raw_value, (int, float)):
        return int(round(raw_value))

    raise TypeError(
        "mutation_num must be an int, float, or percentage string (e.g., '5%')."
    )
