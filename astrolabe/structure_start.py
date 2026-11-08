from __future__ import annotations

import dataclasses
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from astrolabe.core import global_config
from astrolabe.evaluators.utils import ComplexPoses, StandardScores
from astrolabe.stage_transition import (
    TransitionConsistencyGateResult,
    compute_reference_consistency,
)
from astrolabe.utils.mmcif_ids import int_id_to_str_id


from astrolabe.structure_start_config import (
    ReferenceTmScoreConfig, HotspotGateConfig, StructureStartReference, StructureStartConfig,
)

@dataclass(frozen=True)
class StructureStartState:
    config: StructureStartConfig
    reference_scores: StandardScores
    manifest_path: Path
    reference_hotspot_score: float | None = None
    reference_hotspot_threshold: float | None = None


@dataclass(frozen=True)
class ReferenceGateResult:
    gate_active: bool
    score: float
    threshold: float
    passed: bool
    comparisons: dict[str, float]
    per_oligo_tmscore: dict[str, float] = dataclasses.field(default_factory=dict)
    hotspot_score: float | None = None
    hotspot_threshold: float | None = None
    hotspot_passed: bool | None = None
    candidate_count: int = 1
    selected_candidate_index: int = 0


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip() or "reference")


def _source_path_for_alias(alias: str) -> Path:
    try:
        named_files = global_config.named_files
    except RuntimeError as exc:
        raise ValueError("structure_start requires named_files to be registered.") from exc
    parsed_file = (named_files or {}).get(alias)
    if parsed_file is None:
        raise ValueError(f"structure_start reference alias not found: {alias!r}.")
    path = getattr(parsed_file, "file_path", None)
    if not path:
        raise ValueError(f"structure_start named_files[{alias!r}] has no file_path.")
    return Path(path)


def _chain_sequence_for_alias(alias: str, chain_id: str) -> str:
    parsed_file = global_config.named_files.get(alias)
    indices, sequence = parsed_file[str(chain_id)]
    if not indices and not sequence:
        raise ValueError(
            f"structure_start reference alias {alias!r} does not contain chain {chain_id!r}."
        )
    return sequence


def _ordered_input_chains(ref: StructureStartReference) -> list[str]:
    expected_counts = Counter(str(ref.oligo))
    mapped_by_proto: dict[str, list[str]] = defaultdict(list)
    for input_chain, mapped_proto in ref.chain_map.items():
        if mapped_proto not in expected_counts:
            raise ValueError(
                "structure_start reference "
                f"{ref.oligo!r} maps input chain {input_chain!r} to protomer "
                f"{mapped_proto!r}, which is not present in the oligo."
            )
        mapped_by_proto[mapped_proto].append(input_chain)

    for proto, expected_count in expected_counts.items():
        got = mapped_by_proto.get(proto, [])
        if len(got) != expected_count:
            raise ValueError(
                "structure_start reference "
                f"{ref.oligo!r} must map exactly {expected_count} input chain(s) "
                f"to protomer {proto!r}; got {got}."
            )

    ordered: list[str] = []
    cursors: dict[str, int] = {}
    for proto in ref.oligo:
        idx = cursors.get(proto, 0)
        ordered.append(mapped_by_proto[proto][idx])
        cursors[proto] = idx + 1
    return ordered


def _write_normalized_reference_structure(
    *,
    source_path: Path,
    input_chains: list[str],
    output_path: Path,
) -> None:
    from Bio.PDB import MMCIFParser, Model, PDBIO, PDBParser, Structure

    parser = (
        MMCIFParser(QUIET=True)
        if source_path.suffix.lower() in {".cif", ".mmcif"}
        else PDBParser(QUIET=True)
    )
    structure = parser.get_structure("structure_start", str(source_path))
    normalized = Structure.Structure("structure_start")

    for model in structure:
        new_model = Model.Model(model.id)
        available = {str(chain.id): chain for chain in model}
        for idx, input_chain in enumerate(input_chains):
            chain = available.get(str(input_chain))
            if chain is None:
                raise ValueError(
                    f"structure_start requested chain {input_chain!r}, "
                    f"but it is missing from {source_path}."
                )
            copied = chain.copy()
            copied.id = int_id_to_str_id(idx + 1)
            new_model.add(copied)
        normalized.add(new_model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(normalized)
    io.save(str(output_path))


def build_structure_start_state(raw_value: Any, *, loss_computer=None) -> StructureStartState | None:
    config = StructureStartConfig.from_raw(raw_value)
    if config is None:
        return None

    artifact_root = Path(global_config.out_dir) / "structure_start" / "references"
    artifact_root.mkdir(parents=True, exist_ok=True)
    complex_paths: dict[str, Path] = {}
    manifest_refs: dict[str, dict[str, Any]] = {}

    protomers = global_config.protomers
    for oligo, ref in config.references.items():
        missing_protos = [proto for proto in oligo if proto not in protomers]
        if missing_protos:
            raise ValueError(
                f"structure_start reference {oligo!r} uses unknown protomer(s): {missing_protos}."
            )
        input_chains = _ordered_input_chains(ref)
        for input_chain, proto in zip(input_chains, oligo, strict=True):
            ref_seq = _chain_sequence_for_alias(ref.structure_alias, input_chain)
            proto_seq = protomers[proto].sequence
            if len(ref_seq) != len(proto_seq):
                raise ValueError(
                    "structure_start reference sequence length mismatch for "
                    f"{oligo}:{input_chain}->{proto}: reference length={len(ref_seq)} "
                    f"protomer length={len(proto_seq)}."
                )
            if config.sequence_policy == "strict" and ref_seq != proto_seq:
                raise ValueError(
                    "structure_start reference sequence mismatch for "
                    f"{oligo}:{input_chain}->{proto}: reference length={len(ref_seq)} "
                    f"protomer length={len(proto_seq)}."
                )

        source_path = _source_path_for_alias(ref.structure_alias)
        snapshot_path = artifact_root / f"{_safe_label(oligo)}_reference.pdb"
        _write_normalized_reference_structure(
            source_path=source_path,
            input_chains=input_chains,
            output_path=snapshot_path,
        )
        complex_paths[oligo] = snapshot_path
        manifest_refs[oligo] = {
            "structure_alias": ref.structure_alias,
            "source_path": str(source_path),
            "chain_map": dict(ref.chain_map),
            "ordered_input_chains": list(input_chains),
            "normalized_path": str(snapshot_path),
        }

    reference_scores = StandardScores(
        names=list(config.references.keys()),
        plddt={oligo: [1.0] for oligo in config.references.keys()},
        complex_paths=complex_paths,
    )
    if config.hotspot_gate.enabled:
        reference_scores = dataclasses.replace(
            reference_scores,
            complex_poses=ComplexPoses.construct_complex_poses(reference_scores),
        )

    reference_hotspot_score = None
    reference_hotspot_threshold = None
    if config.hotspot_gate.enabled:
        if loss_computer is None:
            raise ValueError("structure_start.hotspot_gate requires a loss computer.")
        reference_hotspot_score = _evaluate_hotspot_expression(
            loss_computer,
            reference_scores,
            config.hotspot_gate.expression,
            context_label="reference",
        )
        reference_hotspot_threshold = max(
            float(config.hotspot_gate.floor),
            float(config.hotspot_gate.reference_fraction)
            * float(reference_hotspot_score)
            - float(config.hotspot_gate.tolerance),
        )

    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "mode": config.mode,
                "gate_timing": list(config.gate_timing),
                "aggregate": config.aggregate,
                "sequence_policy": config.sequence_policy,
                "reference_tmscore": dataclasses.asdict(config.reference_tmscore),
                "hotspot_gate": dataclasses.asdict(config.hotspot_gate),
                "reference_hotspot_score": reference_hotspot_score,
                "reference_hotspot_threshold": reference_hotspot_threshold,
                "references": manifest_refs,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return StructureStartState(
        config=config,
        reference_scores=reference_scores,
        manifest_path=manifest_path,
        reference_hotspot_score=reference_hotspot_score,
        reference_hotspot_threshold=reference_hotspot_threshold,
    )


def _evaluate_hotspot_expression(
    loss_computer,
    scores: StandardScores,
    expression: str,
    *,
    context_label: str,
) -> float:
    try:
        env = loss_computer.score_context_for_expressions(scores, [expression])
        value = loss_computer.weighted_score(env, expression)
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate structure_start.hotspot_gate.expression "
            f"{expression!r} on {context_label} structures."
        ) from exc
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"structure_start.hotspot_gate.expression {expression!r} on "
            f"{context_label} structures did not produce a numeric value: {value!r}."
        ) from exc


def _per_oligo_tmscore(comparisons: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in comparisons.items():
        if not str(key).startswith("min~"):
            continue
        oligo = str(key).split("~", 1)[1]
        out[oligo] = float(value)
    return out


def evaluate_structure_start_gate(
    eval_result: StandardScores,
    *,
    timing: str,
    loss_computer=None,
    compare_fn=None,
    candidate_count: int = 1,
    selected_candidate_index: int = 0,
) -> ReferenceGateResult | None:
    try:
        state = global_config.structure_start
    except RuntimeError:
        return None
    if state is None:
        return None
    if timing not in state.config.gate_timing:
        return None

    tm_result = TransitionConsistencyGateResult(
        gate_active=True,
        score=1.0,
        threshold=0.0,
        passed=True,
        comparisons={},
    )
    if state.config.reference_tmscore.enabled:
        tm_result = compute_reference_consistency(
            current_scores=eval_result,
            reference_scores=state.reference_scores,
            threshold=state.config.reference_tmscore.threshold,
            compare_fn=compare_fn,
        )

    hotspot_score = None
    hotspot_threshold = state.reference_hotspot_threshold
    hotspot_passed = None
    if state.config.hotspot_gate.enabled:
        if loss_computer is None:
            raise ValueError("structure_start hotspot gate requires a loss computer.")
        hotspot_score = _evaluate_hotspot_expression(
            loss_computer,
            eval_result,
            state.config.hotspot_gate.expression,
            context_label="candidate",
        )
        hotspot_passed = bool(hotspot_score >= float(hotspot_threshold))

    hard_gate = state.config.mode == "repredict_gate"
    passed = bool(tm_result.passed)
    if hotspot_passed is not None:
        passed = passed and bool(hotspot_passed)

    return ReferenceGateResult(
        gate_active=hard_gate,
        score=float(tm_result.score),
        threshold=float(tm_result.threshold),
        passed=passed,
        comparisons={k: float(v) for k, v in tm_result.comparisons.items()},
        per_oligo_tmscore=_per_oligo_tmscore(tm_result.comparisons),
        hotspot_score=hotspot_score,
        hotspot_threshold=hotspot_threshold,
        hotspot_passed=hotspot_passed,
        candidate_count=int(candidate_count),
        selected_candidate_index=int(selected_candidate_index),
    )


def select_structure_start_candidate(
    eval_result: StandardScores,
    *,
    timing: str,
    loss_computer=None,
    compare_fn=None,
) -> tuple[StandardScores, ReferenceGateResult | None]:
    try:
        state = global_config.structure_start
    except RuntimeError:
        state = None
    if state is None or state.config.mode != "soft_select":
        return eval_result, evaluate_structure_start_gate(
            eval_result,
            timing=timing,
            loss_computer=loss_computer,
            compare_fn=compare_fn,
        )

    candidates = (
        getattr(eval_result, "candidate_scores", None)
        or getattr(eval_result, "candidate_results", None)
        or None
    )
    if not candidates:
        result = evaluate_structure_start_gate(
            eval_result,
            timing=timing,
            loss_computer=loss_computer,
            compare_fn=compare_fn,
            candidate_count=1,
            selected_candidate_index=0,
        )
        return eval_result, result

    scored: list[tuple[float, int, StandardScores, ReferenceGateResult | None]] = []
    for idx, candidate in enumerate(candidates):
        result = evaluate_structure_start_gate(
            candidate,
            timing=timing,
            loss_computer=loss_computer,
            compare_fn=compare_fn,
            candidate_count=len(candidates),
            selected_candidate_index=idx,
        )
        score = float(getattr(result, "score", 0.0) if result is not None else 0.0)
        scored.append((score, idx, candidate, result))
    scored.sort(key=lambda item: item[0], reverse=True)
    _score, idx, selected, result = scored[0]
    if result is not None:
        result = dataclasses.replace(
            result,
            candidate_count=len(candidates),
            selected_candidate_index=idx,
        )
    return selected, result


def reference_gate_score_context(result: ReferenceGateResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    context: dict[str, Any] = {
        "reference_tmscore": float(result.score),
        "reference_gate_passed": 1.0 if bool(result.passed) else 0.0,
        "reference_tmscore_threshold": float(result.threshold),
        "reference_soft_select_candidate_count": int(result.candidate_count),
        "reference_soft_select_selected_index": int(result.selected_candidate_index),
    }
    if result.hotspot_score is not None:
        context["reference_hotspot_score"] = float(result.hotspot_score)
    if result.hotspot_threshold is not None:
        context["reference_hotspot_threshold"] = float(result.hotspot_threshold)
    if result.hotspot_passed is not None:
        context["reference_hotspot_passed"] = 1.0 if result.hotspot_passed else 0.0
    for oligo, score in result.per_oligo_tmscore.items():
        context.setdefault(oligo, {})["reference_tmscore"] = float(score)
    return context
