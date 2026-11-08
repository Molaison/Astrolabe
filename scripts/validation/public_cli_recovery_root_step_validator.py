#!/usr/bin/env python3
"""Read-only validator for the exact root-step metadata exception.

The raw comparison remains ISSUES. This tool can report PASS only when each
selected public-CLI case differs by exactly:

  standard_scores.json[0].step: null -> 0
  rng_trace.json[0].step:        null -> 0

Every other scientific, structural, accounting, source, weight, and GPU field
must match the case's native reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


DEFAULT_SOURCE_FINGERPRINT = (
    "bee589e36e30e4391cd4164148b365215002bc2897bec621e5519607b762f63c"
)
EXPECTED_TRUE_FIELDS = (
    "exact_beam_selection_match",
    "exact_decision_projection_match",
    "exact_numeric_match",
    "exact_rng_trace_match",
    "exact_sequence_and_weighted_score_match",
    "exact_structure_atom_identity_match",
    "exact_structure_coordinate_match",
    "same_gpu",
    "same_scientific_config_except_execution",
    "same_weights",
)
EXPECTED_FALSE_FIELDS = (
    "exact_rng_trace_detail_match",
    "exact_score_field_topology_match",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return (math.isnan(left) and math.isnan(right)) or left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _json_differences(
    left: Any,
    right: Any,
    *,
    path: str = "$",
    output: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    output = [] if output is None else output
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) - set(right)):
            output.append(
                {
                    "path": f"{path}.{key}",
                    "kind": "left-only",
                    "left": left[key],
                    "right": None,
                }
            )
        for key in sorted(set(right) - set(left)):
            output.append(
                {
                    "path": f"{path}.{key}",
                    "kind": "right-only",
                    "left": None,
                    "right": right[key],
                }
            )
        for key in sorted(set(left) & set(right)):
            _json_differences(
                left[key],
                right[key],
                path=f"{path}.{key}",
                output=output,
            )
        return output
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            output.append(
                {
                    "path": path,
                    "kind": "length",
                    "left": len(left),
                    "right": len(right),
                }
            )
        for index, (a, b) in enumerate(zip(left, right)):
            _json_differences(
                a,
                b,
                path=f"{path}[{index}]",
                output=output,
            )
        return output
    if not _equal(left, right):
        output.append(
            {
                "path": path,
                "kind": "value",
                "left": left,
                "right": right,
            }
        )
    return output


def _without_artifact_paths(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$path"}:
            return {"$path": "<artifact-path>"}
        return {
            key: _without_artifact_paths(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_without_artifact_paths(item) for item in value]
    return value


def is_allowed_root_step_difference(
    differences: Sequence[dict[str, Any]],
) -> bool:
    return (
        len(differences) == 1
        and differences[0].get("path") == "$[0].step"
        and differences[0].get("kind") == "value"
        and differences[0].get("left", object()) is None
        and type(differences[0].get("right")) is int
        and differences[0].get("right") == 0
    )


def _compare_npz(left_path: Path, right_path: Path) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    with np.load(left_path, allow_pickle=False) as left, np.load(
        right_path,
        allow_pickle=False,
    ) as right:
        left_keys = set(left.files)
        right_keys = set(right.files)
        for key in sorted(left_keys - right_keys):
            failures.append({"key": key, "reason": "left-only"})
        for key in sorted(right_keys - left_keys):
            failures.append({"key": key, "reason": "right-only"})
        for key in sorted(left_keys & right_keys):
            a = left[key]
            b = right[key]
            if a.dtype != b.dtype:
                failures.append(
                    {
                        "key": key,
                        "reason": "dtype",
                        "left": a.dtype.str,
                        "right": b.dtype.str,
                    }
                )
            elif a.shape != b.shape:
                failures.append(
                    {
                        "key": key,
                        "reason": "shape",
                        "left": list(a.shape),
                        "right": list(b.shape),
                    }
                )
            elif not np.array_equal(a, b, equal_nan=True):
                failures.append({"key": key, "reason": "value"})
    return {
        "equal": not failures,
        "array_count_left": len(left_keys),
        "array_count_right": len(right_keys),
        "left_sha256": _sha256(left_path),
        "right_sha256": _sha256(right_path),
        "failures": failures,
    }


def validate_case(
    attempt: Path,
    case_name: str,
    *,
    expected_source_fingerprint: str,
) -> dict[str, Any]:
    case_root = attempt / "cases" / case_name
    comparison_path = case_root / "comparison.json"
    result_path = case_root / "result.json"
    for path in (comparison_path, result_path):
        if not path.is_file():
            raise ValueError(f"missing case artifact: {path}")
    comparison = _load_json(comparison_path)
    result = _load_json(result_path)
    reference_name = comparison.get("reference")
    if not isinstance(reference_name, str) or not reference_name:
        raise ValueError(f"case {case_name!r} has no comparison reference")
    reference_root = attempt / "references" / reference_name
    if not (reference_root / "result.json").is_file():
        # Fresh baselines retain their physical directory while comparison
        # records use the logical native case name. Resolve only the exact
        # reference recorded by this attempt, never a neighbouring run.
        summary = _load_json(attempt / "summary.json")
        record = summary["baseline_decisions"][case_name]
        recorded = Path(record["result"])
        if (
            len(recorded.parts) < 3
            or recorded.parts[-3] != "references"
            or recorded.parts[-1] != "result.json"
            or recorded.parts[-2] in {".", ".."}
        ):
            raise ValueError("invalid recorded fresh baseline result path")
        reference_root = attempt / "references" / recorded.parts[-2]
    reference_result_path = reference_root / "result.json"
    if not reference_result_path.is_file():
        raise ValueError(f"missing reference result: {reference_result_path}")
    reference_result = _load_json(reference_result_path)
    if reference_result.get("case") != reference_name:
        raise ValueError("recorded baseline differs from comparison reference")
    case_evidence = case_root / "evidence"
    reference_evidence = reference_root / "evidence"

    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks[name] = {
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        }

    source_manifest = _load_json(attempt / "source_manifest.json")
    check(
        "source_fingerprint",
        source_manifest.get("fingerprint") == expected_source_fingerprint,
        {"fingerprint": source_manifest.get("fingerprint")},
    )
    check(
        "raw_comparison_issues_preserved",
        comparison.get("status") == "ISSUES",
        {"status": comparison.get("status")},
    )
    check(
        "comparison_scope",
        comparison.get("case") == case_name,
        {
            "case": comparison.get("case"),
            "reference": reference_name,
        },
    )
    config_differences = list(comparison.get("config_differences") or ())
    check(
        "scientific_config_exact_except_execution",
        not config_differences,
        {"differences": config_differences},
    )
    true_fields = {
        field: comparison.get(field) for field in EXPECTED_TRUE_FIELDS
    }
    false_fields = {
        field: comparison.get(field) for field in EXPECTED_FALSE_FIELDS
    }
    check(
        "comparison_flags_exact",
        all(value is True for value in true_fields.values())
        and all(value is False for value in false_fields.values()),
        {
            "expected_true": true_fields,
            "expected_false": false_fields,
        },
    )

    score_arrays = _compare_npz(
        reference_evidence / "standard_scores.npz",
        case_evidence / "standard_scores.npz",
    )
    rng_arrays = _compare_npz(
        reference_evidence / "rng_trace.npz",
        case_evidence / "rng_trace.npz",
    )
    coordinate_arrays = _compare_npz(
        reference_evidence / "structure_coordinates.npz",
        case_evidence / "structure_coordinates.npz",
    )
    check("score_arrays_exact", score_arrays["equal"], score_arrays)
    check("rng_arrays_exact", rng_arrays["equal"], rng_arrays)
    check(
        "structure_coordinate_arrays_exact",
        coordinate_arrays["equal"],
        coordinate_arrays,
    )

    score_differences = _json_differences(
        _without_artifact_paths(
            _load_json(reference_evidence / "standard_scores.json")
        ),
        _without_artifact_paths(
            _load_json(case_evidence / "standard_scores.json")
        ),
    )
    rng_differences = _json_differences(
        _load_json(reference_evidence / "rng_trace.json"),
        _load_json(case_evidence / "rng_trace.json"),
    )
    check(
        "score_json_only_root_step",
        is_allowed_root_step_difference(score_differences),
        {"differences": score_differences},
    )
    check(
        "rng_json_only_root_step",
        is_allowed_root_step_difference(rng_differences),
        {"differences": rng_differences},
    )
    check(
        "exactly_two_root_step_metadata_differences",
        is_allowed_root_step_difference(score_differences)
        and is_allowed_root_step_difference(rng_differences),
        {
            "score_json_path": "standard_scores.json[0].step",
            "rng_json_path": "rng_trace.json[0].step",
        },
    )

    projection = {
        "sequences_equal": _equal(
            reference_result.get("sequences"),
            result.get("sequences"),
        ),
        "weighted_scores_equal": _equal(
            reference_result.get("weighted_scores"),
            result.get("weighted_scores"),
        ),
        "decision_kind_equal": (
            reference_result.get("decision_kind")
            == result.get("decision_kind")
        ),
    }
    check(
        "result_projection_exact",
        all(projection.values()),
        projection,
    )

    accounting = result.get("resume_accounting") or {}
    accounting_exact = (
        result.get("status") == "evidence-complete"
        and result.get("model_calls") == 2
        and accounting.get("cut_model_requests") == 1
        and accounting.get("resume_model_requests") == 1
        and accounting.get("duplicate_model_requests") == 0
        and accounting.get("cut_steps") == [0]
        and accounting.get("resume_steps") == [1]
    )
    check(
        "request_accounting_exact",
        accounting_exact,
        {
            "status": result.get("status"),
            "model_calls": result.get("model_calls"),
            "accounting": accounting,
        },
    )

    status = (
        "PASS"
        if all(item["status"] == "PASS" for item in checks.values())
        else "ISSUES"
    )
    return {
        "schema": "astrolabe.public-cli-recovery.root-step-case-validation/v1",
        "case": case_name,
        "reference": reference_name,
        "status": status,
        "raw_comparison_status": comparison.get("status"),
        "allowed_exception": {
            "paths": [
                "standard_scores.json[0].step",
                "rng_trace.json[0].step",
            ],
            "reference": None,
            "case": 0,
            "scope": "root evaluation index 0 only",
        },
        "checks": checks,
        "artifact_hashes": {
            "comparison_sha256": _sha256(comparison_path),
            "result_sha256": _sha256(result_path),
            "score_npz_sha256": _sha256(
                case_evidence / "standard_scores.npz"
            ),
            "rng_npz_sha256": _sha256(case_evidence / "rng_trace.npz"),
            "structure_npz_sha256": _sha256(
                case_evidence / "structure_coordinates.npz"
            ),
        },
    }


def validate_attempt(
    attempt: Path,
    *,
    cases: Sequence[str],
    expected_source_fingerprint: str = DEFAULT_SOURCE_FINGERPRINT,
) -> dict[str, Any]:
    case_results = {
        name: validate_case(
            attempt,
            name,
            expected_source_fingerprint=expected_source_fingerprint,
        )
        for name in cases
    }
    status = (
        "PASS"
        if case_results
        and all(item["status"] == "PASS" for item in case_results.values())
        else "ISSUES"
    )
    return {
        "schema": "astrolabe.public-cli-recovery.root-step-attempt-validation/v1",
        "attempt": str(attempt),
        "source_fingerprint": expected_source_fingerprint,
        "status": status,
        "raw_comparison_statuses_preserved": {
            name: item["raw_comparison_status"]
            for name, item in case_results.items()
        },
        "cases": case_results,
    }


def _discover_cases(attempt: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.parent.name
            for path in (attempt / "cases").glob("*/comparison.json")
            if path.is_file()
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only validation of only the two root-step metadata "
            "normalization exceptions."
        )
    )
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument(
        "--expected-source-fingerprint",
        default=DEFAULT_SOURCE_FINGERPRINT,
    )
    arguments = parser.parse_args(argv)
    attempt = arguments.attempt.expanduser().resolve()
    cases = tuple(arguments.cases or _discover_cases(attempt))
    if not cases:
        print(
            json.dumps(
                {
                    "status": "ISSUES",
                    "error": f"no completed case comparisons under {attempt}",
                },
                sort_keys=True,
            )
        )
        return 1
    try:
        result = validate_attempt(
            attempt,
            cases=cases,
            expected_source_fingerprint=arguments.expected_source_fingerprint,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "ISSUES",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
