"""Pure structure-start configuration schema for early validation."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
from omegaconf import OmegaConf

@dataclass(frozen=True)
class ReferenceTmScoreConfig:
    enabled: bool = True
    threshold: float = 0.85


@dataclass(frozen=True)
class HotspotGateConfig:
    enabled: bool = False
    expression: str = "pair_hotspot_support"
    floor: float = 0.25
    reference_fraction: float = 0.8
    tolerance: float = 0.05


@dataclass(frozen=True)
class StructureStartReference:
    oligo: str
    structure_alias: str
    chain_map: dict[str, str]


@dataclass(frozen=True)
class StructureStartConfig:
    mode: str
    gate_timing: tuple[str, ...]
    aggregate: str
    references: dict[str, StructureStartReference]
    reference_tmscore: ReferenceTmScoreConfig
    hotspot_gate: HotspotGateConfig
    sequence_policy: str = "strict"

    @classmethod
    def from_raw(cls, raw_value: Any) -> "StructureStartConfig | None":
        if raw_value is None:
            return None
        raw = _to_plain(raw_value)
        if raw in ({}, "null", "None"):
            return None
        if not isinstance(raw, dict):
            raise ValueError("structure_start must be a mapping.")

        allowed = {
            "mode",
            "gate_timing",
            "aggregate",
            "sequence_policy",
            "references",
            "reference_tmscore",
            "hotspot_gate",
        }
        unknown = sorted(set(raw.keys()) - allowed)
        if unknown:
            raise ValueError(f"structure_start has unsupported fields: {unknown}.")

        mode = str(raw.get("mode", "repredict_gate")).strip()
        if mode not in {"repredict_gate", "soft_select"}:
            raise ValueError(
                "structure_start.mode must be one of: repredict_gate, soft_select."
            )

        timings_raw = raw.get("gate_timing", ["step0", "pre_accept"])
        timings = _as_string_tuple(timings_raw, field="structure_start.gate_timing")
        allowed_timings = {"step0", "pre_accept", "stage_transition"}
        bad_timings = sorted(set(timings) - allowed_timings)
        if bad_timings:
            raise ValueError(
                f"structure_start.gate_timing has unsupported values: {bad_timings}."
            )

        aggregate = str(raw.get("aggregate", "min")).strip()
        if aggregate != "min":
            raise ValueError("structure_start.aggregate currently supports only 'min'.")

        sequence_policy = str(raw.get("sequence_policy", "strict")).strip()
        if sequence_policy not in {"strict", "length_only"}:
            raise ValueError(
                "structure_start.sequence_policy must be one of: strict, length_only."
            )

        references_raw = raw.get("references")
        if not isinstance(references_raw, dict) or not references_raw:
            raise ValueError("structure_start.references must be a non-empty mapping.")
        references: dict[str, StructureStartReference] = {}
        for oligo_raw, ref_raw in references_raw.items():
            oligo = str(oligo_raw).strip()
            if not oligo:
                raise ValueError("structure_start.references contains an empty oligo.")
            if not isinstance(ref_raw, dict):
                raise ValueError(f"structure_start.references.{oligo} must be a mapping.")
            ref_unknown = sorted(set(ref_raw.keys()) - {"structure_alias", "chain_map"})
            if ref_unknown:
                raise ValueError(
                    f"structure_start.references.{oligo} has unsupported fields: {ref_unknown}."
                )
            structure_alias = str(ref_raw.get("structure_alias", "")).strip()
            if not structure_alias:
                raise ValueError(
                    f"structure_start.references.{oligo}.structure_alias is required."
                )
            chain_map_raw = ref_raw.get("chain_map")
            if not isinstance(chain_map_raw, dict) or not chain_map_raw:
                raise ValueError(
                    f"structure_start.references.{oligo}.chain_map must be a non-empty mapping."
                )
            chain_map = {
                str(input_chain).strip(): str(proto).strip()
                for input_chain, proto in chain_map_raw.items()
            }
            if any(not k or not v for k, v in chain_map.items()):
                raise ValueError(
                    f"structure_start.references.{oligo}.chain_map cannot contain empty keys/values."
                )
            references[oligo] = StructureStartReference(
                oligo=oligo,
                structure_alias=structure_alias,
                chain_map=chain_map,
            )

        tm_raw = raw.get("reference_tmscore") or {}
        if not isinstance(tm_raw, dict):
            raise ValueError("structure_start.reference_tmscore must be a mapping.")
        tm_unknown = sorted(set(tm_raw.keys()) - {"enabled", "threshold"})
        if tm_unknown:
            raise ValueError(
                f"structure_start.reference_tmscore has unsupported fields: {tm_unknown}."
            )
        tm_threshold = _float_in_unit_interval(
            tm_raw.get("threshold", 0.85),
            field="structure_start.reference_tmscore.threshold",
        )
        reference_tmscore = ReferenceTmScoreConfig(
            enabled=bool(tm_raw.get("enabled", True)),
            threshold=tm_threshold,
        )

        hotspot_raw = raw.get("hotspot_gate") or {}
        if not isinstance(hotspot_raw, dict):
            raise ValueError("structure_start.hotspot_gate must be a mapping.")
        hotspot_unknown = sorted(
            set(hotspot_raw.keys())
            - {"enabled", "expression", "floor", "reference_fraction", "tolerance"}
        )
        if hotspot_unknown:
            raise ValueError(
                f"structure_start.hotspot_gate has unsupported fields: {hotspot_unknown}."
            )
        hotspot_gate = HotspotGateConfig(
            enabled=bool(hotspot_raw.get("enabled", False)),
            expression=str(hotspot_raw.get("expression", "pair_hotspot_support")).strip()
            or "pair_hotspot_support",
            floor=_float_in_unit_interval(
                hotspot_raw.get("floor", 0.25),
                field="structure_start.hotspot_gate.floor",
            ),
            reference_fraction=_float_in_unit_interval(
                hotspot_raw.get("reference_fraction", 0.8),
                field="structure_start.hotspot_gate.reference_fraction",
            ),
            tolerance=_float_in_unit_interval(
                hotspot_raw.get("tolerance", 0.05),
                field="structure_start.hotspot_gate.tolerance",
            ),
        )

        return cls(
            mode=mode,
            gate_timing=timings,
            aggregate=aggregate,
            references=references,
            reference_tmscore=reference_tmscore,
            hotspot_gate=hotspot_gate,
            sequence_policy=sequence_policy,
        )



def _to_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _as_string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    value = _to_plain(value)
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError(f"{field} must be a string or list of strings.")
    out = tuple(str(item).strip() for item in items if str(item).strip())
    if not out:
        raise ValueError(f"{field} cannot be empty.")
    return out


def _float_in_unit_interval(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{field} must be within [0, 1].")
    return parsed
