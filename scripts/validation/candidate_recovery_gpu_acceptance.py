#!/usr/bin/env python3
"""Bounded real-GPU acceptance for durable candidate collection recovery.

The default action only prepares an immutable plan.  ``--execute`` runs one
uninterrupted four-request candidate batch and one four-request committed-round
cut/recovery batch.  Evaluator instrumentation enforces two calls per child and
therefore an eight-request hard cap across the whole acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
UNIFIED_PATH = Path(__file__).with_name("unified_gpu_acceptance.py")
SCHEMA = "astrolabe.candidate-recovery-gpu-acceptance/v1"
BASE_SEED = 16
EXPECTED_CHILD_SEEDS = {"cand_0001": 17, "cand_0002": 18}
CANDIDATE_COUNT = 2
MIN_COMPLETED = 2
DECISION_ROUNDS = 1
CANDIDATE_STEPS = 2
MAX_IN_FLIGHT = 2
REQUESTS_PER_CHILD = 2
REQUESTS_PER_SCENARIO = CANDIDATE_COUNT * REQUESTS_PER_CHILD
TOTAL_REQUEST_BUDGET = 2 * REQUESTS_PER_SCENARIO
VALIDATION_MAX_WAIT_S = 900.0
_MODEL_GROUP_TIMESTAMP = re.compile(
    r'^(?P<head>_ma_model_list\.model_group_name[ \t]+".*? @ )'
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r'(?P<tail>\)"[ \t]*)(?P<newline>\r?\n)?$'
)


def _load_unified():
    name = "astrolabe_candidate_recovery_unified"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, UNIFIED_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load unified acceptance helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SHARED = _load_unified()


def _source_manifest() -> dict[str, str]:
    manifest = dict(SHARED._read_source_manifest().files)
    current = Path(__file__).resolve()
    manifest[current.relative_to(REPO_ROOT).as_posix()] = SHARED._sha256(current)
    return manifest


def _source_fingerprint(manifest: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _candidate_plan(
    *,
    out_root: Path,
    profile: Path,
    max_wait_s: float,
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    if max_wait_s <= 0:
        raise ValueError("--max-wait-s must be positive")
    if out_root.exists() and any(out_root.iterdir()):
        raise SHARED.AcceptanceError(
            f"acceptance output is not empty: {out_root}"
        )
    out_root.mkdir(parents=True, exist_ok=True)
    instrumentation = out_root / "instrumentation"
    instrumentation.mkdir()
    _atomic_text(
        instrumentation / "sitecustomize.py",
        SHARED._SITECUSTOMIZE,
    )
    configs: dict[str, str] = {}
    scenarios = ("uninterrupted", "committed_cut")
    for scenario in scenarios:
        scenario_root = out_root / scenario
        scenario_root.mkdir()
        run_root = scenario_root / "run"
        cfg = SHARED._build_config(
            profile=profile,
            fixture="single-20aa",
            out_dir=run_root,
            task_name=f"candidate_recovery_{scenario}",
            execution="daemon",
            generic_search=False,
            seed=BASE_SEED,
        )
        config_path = scenario_root / "config.yaml"
        SHARED._save_standalone(cfg, config_path)
        configs[scenario] = str(config_path)
    manifest = _source_manifest()
    fingerprint = _source_fingerprint(manifest)
    SHARED._atomic_json(
        out_root / "source_manifest.json",
        {
            "schema": f"{SCHEMA}.source-manifest",
            "fingerprint": fingerprint,
            "files": manifest,
        },
    )
    plan = {
        "schema": f"{SCHEMA}.plan",
        "status": "prepared",
        "source_fingerprint": fingerprint,
        "profile": str(profile),
        "profile_sha256": SHARED._sha256(profile),
        "fixture": "single-20aa",
        "execution": "daemon",
        "configs": configs,
        "base_seed": BASE_SEED,
        "expected_child_seeds": EXPECTED_CHILD_SEEDS,
        "num_candidates": CANDIDATE_COUNT,
        "min_completed": MIN_COMPLETED,
        "decision_rounds": DECISION_ROUNDS,
        "candidate_steps": CANDIDATE_STEPS,
        "max_in_flight": MAX_IN_FLIGHT,
        "max_wait_s": float(max_wait_s),
        "requests_per_child_cap": REQUESTS_PER_CHILD,
        "requests_per_scenario_cap": REQUESTS_PER_SCENARIO,
        "total_request_budget": TOTAL_REQUEST_BUDGET,
        "capture": {
            "per_child": True,
            "capture_root": str(out_root / "<scenario>" / "captures"),
            "claims_isolated_per_child": True,
        },
        "cut": {
            "scenario": "committed_cut",
            "signal": "SIGKILL parent after collection/commit checkpoint",
            "resume": True,
            "retry_issued_without_terminal": False,
        },
        "af3": {
            "model_dir": SHARED.AF3_MODEL_DIR,
            "num_seeds": 1,
            "num_recycles": 1,
            "num_samples": 1,
            "weighted_score_expression": "plddt",
            "mutation_num": 1,
        },
        "launch_policy": "prepare by default; --execute is required",
    }
    SHARED._atomic_json(out_root / "plan.json", plan)
    return plan


def _load_prepared_plan(
    *,
    out_root: Path,
    profile: Path,
    max_wait_s: float,
) -> dict[str, Any]:
    path = out_root / "plan.json"
    if not path.is_file():
        raise SHARED.AcceptanceError(
            f"--execute requires a prepared plan at {path}"
        )
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != f"{SCHEMA}.plan":
        raise SHARED.AcceptanceError("unsupported candidate recovery plan schema")
    if Path(plan.get("profile", "")).resolve() != profile:
        raise SHARED.AcceptanceError("candidate recovery profile changed")
    if float(plan.get("max_wait_s")) != float(max_wait_s):
        raise SHARED.AcceptanceError("candidate recovery max_wait_s changed")
    return plan


def _scenario_environment(
    plan: Mapping[str, Any],
    scenario: str,
) -> dict[str, str]:
    out_root = Path(plan["configs"][scenario]).parent.parent
    instrumentation = out_root / "instrumentation"
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(instrumentation),
            str(REPO_ROOT),
            existing or "",
        )
        if part
    )
    env["ASTROLABE_ACCEPTANCE_CAPTURE_ROOT"] = str(
        out_root / scenario / "captures"
    )
    env["ASTROLABE_ACCEPTANCE_MAX_CALLS"] = str(REQUESTS_PER_CHILD)
    env["ASTROLABE_RUNTIME_ACTIVE"] = "1"
    env["PYTHONHASHSEED"] = str(EXPECTED_CHILD_SEEDS["cand_0001"])
    return env


def _command(
    *,
    config: Path,
    run_root: Path,
    max_wait_s: float,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "astrolabe.cli",
        "experimental",
        "candidate-batch",
        "--config-yaml",
        str(config),
        "--out-root",
        str(run_root),
        "--runner",
        str(REPO_ROOT / "scripts" / "astrolabe"),
        "--num-candidates",
        str(CANDIDATE_COUNT),
        "--min-completed",
        str(MIN_COMPLETED),
        "--max-in-flight",
        str(MAX_IN_FLIGHT),
        "--max-wait-s",
        str(max_wait_s),
        "--decision-rounds",
        str(DECISION_ROUNDS),
        "--candidate-steps",
        str(CANDIDATE_STEPS),
        "--format",
        "json",
    ]
    if resume:
        command.append("--resume")
    return command


def _launch(
    *,
    command: list[str],
    cwd: Path,
    env: Mapping[str, str],
    stdout_log: Path,
    stderr_log: Path,
) -> subprocess.Popen:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stdout = stdout_log.open("w", encoding="utf-8")
    stderr = stderr_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=dict(env),
        stdout=stdout,
        stderr=stderr,
        text=True,
        start_new_session=True,
    )
    stdout.close()
    stderr.close()
    return process


def _committed(journal: Path) -> bool:
    if not journal.is_file():
        return False
    try:
        with sqlite3.connect(f"file:{journal}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT 1 FROM records WHERE kind='checkpoint' "
                "AND name LIKE 'collection/commit/%' LIMIT 1"
            ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False


def _parse_json_output(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SHARED.AcceptanceError(f"empty candidate summary: {path}")
    return json.loads(text)


def _read_receipt(run_root: Path, job_id: str):
    from astrolabe.experimental.candidate_batch_runner import (
        _read_candidate_receipt,
    )

    return _read_candidate_receipt(
        run_root / "candidate_pool" / job_id / "candidate_receipt.json"
    )


def _capture_evidence(
    *,
    capture_root: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for child_root in sorted(capture_root.iterdir()):
        capture = child_root / "capture.pickle"
        if not capture.is_file():
            raise SHARED.AcceptanceError(
                f"missing child capture for {child_root.name}: {capture}"
            )
        records = SHARED._read_capture_records(capture)
        if len(records) != REQUESTS_PER_CHILD:
            raise SHARED.AcceptanceError(
                f"{child_root.name} made {len(records)} evaluator requests, "
                f"expected {REQUESTS_PER_CHILD}"
            )
        if any(record.get("status") != "ok" for record in records):
            raise SHARED.AcceptanceError(
                f"{child_root.name} capture contains a failed evaluator request"
            )
        child_evidence = evidence_root / child_root.name
        arrays, scores, rng_arrays, rng, coordinates, structures = (
            SHARED._score_and_coordinate_evidence(child_evidence, records)
        )
        SHARED._write_npz(child_evidence / "standard_scores.npz", arrays)
        SHARED._write_npz(child_evidence / "rng_trace.npz", rng_arrays)
        SHARED._write_npz(
            child_evidence / "structure_coordinates.npz",
            coordinates,
        )
        SHARED._atomic_json(
            child_evidence / "standard_scores.json",
            scores,
        )
        SHARED._atomic_json(child_evidence / "rng_trace.json", rng)
        SHARED._atomic_json(
            child_evidence / "structure_coordinates.json",
            structures,
        )
        evidence[child_root.name] = {
            "capture": str(capture),
            "capture_sha256": SHARED._sha256(capture),
            "request_count": len(records),
            "evidence": str(child_evidence),
        }
    if set(evidence) != set(EXPECTED_CHILD_SEEDS):
        raise SHARED.AcceptanceError(
            "candidate capture identities differ from the fixed child set"
        )
    return evidence


def _validate_scenario(
    *,
    scenario: str,
    run_root: Path,
    summary: Mapping[str, Any],
    capture_root: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    receipts = {
        job_id: _read_receipt(run_root, job_id)
        for job_id in EXPECTED_CHILD_SEEDS
    }
    for job_id, receipt in receipts.items():
        if receipt.status != "completed":
            raise SHARED.AcceptanceError(
                f"{scenario}/{job_id} is not completed: {receipt.status}"
            )
        if receipt.actual_seed != EXPECTED_CHILD_SEEDS[job_id]:
            raise SHARED.AcceptanceError(
                f"{scenario}/{job_id} changed its expected child seed"
            )
        if receipt.result is None or not receipt.output_manifest:
            raise SHARED.AcceptanceError(
                f"{scenario}/{job_id} lacks typed result or output manifest"
            )
        if not receipt.cli_provenance or receipt.cli_provenance.get(
            "returncode"
        ) != 0:
            raise SHARED.AcceptanceError(
                f"{scenario}/{job_id} lacks completed CLI provenance"
            )
        final = receipt.final_config
        af3 = final["evaluator"]["alphafold3"]
        term = final["evaluator"]["evaluator_terms"]["Alphafold3"]
        if (
            int(final["steps"]) != CANDIDATE_STEPS
            or int(final["mutation_num"]) != 1
            or final["stages"]["default"]["mutators"] != ["MCMC"]
            or final["stages"]["default"]["weighted_score_expression"]
            != "plddt"
            or int(af3["num_seeds"]) != 1
            or int(af3["num_recycles"]) != 1
            or int(af3["num_samples"]) != 1
            or int(term["num_seeds"]) != 1
            or int(term["num_recycles"]) != 1
            or int(term["num_samples"]) != 1
        ):
            raise SHARED.AcceptanceError(
                f"{scenario}/{job_id} changed the bounded AF3 configuration"
            )
        sequence = str(final["seqs"]["A"]).strip().lstrip("+").split("@", 1)[0]
        if len(sequence) != 20:
            raise SHARED.AcceptanceError(
                f"{scenario}/{job_id} is not the fixed 20-residue fixture"
            )
    collections = summary.get("collections", [])
    if not collections or collections[0].get("trigger") != "min_completed":
        raise SHARED.AcceptanceError(
            f"{scenario} did not commit a min_completed collection"
        )
    evidence = _capture_evidence(
        capture_root=capture_root,
        evidence_root=evidence_root,
    )
    request_count = sum(item["request_count"] for item in evidence.values())
    if request_count != REQUESTS_PER_SCENARIO:
        raise SHARED.AcceptanceError(
            f"{scenario} used {request_count} requests, expected "
            f"{REQUESTS_PER_SCENARIO}"
        )
    return {
        "status": "passed",
        "run_root": str(run_root),
        "receipts": {
            job_id: {
                "seed": receipt.actual_seed,
                "content_id": receipt.content_id,
                "status": receipt.status,
                "output_manifest": list(receipt.output_manifest),
            }
            for job_id, receipt in receipts.items()
        },
        "collections": collections,
        "selected_candidate_ids": summary.get("selected_candidate_ids"),
        "request_count": request_count,
        "capture_evidence": evidence,
    }


def _run_uninterrupted(
    plan: Mapping[str, Any],
    *,
    scenario: str,
) -> dict[str, Any]:
    config = Path(plan["configs"][scenario])
    scenario_root = config.parent
    run_root = scenario_root / "run"
    stdout = scenario_root / "stdout.log"
    stderr = scenario_root / "stderr.log"
    env = _scenario_environment(plan, scenario)
    process = _launch(
        command=_command(
            config=config,
            run_root=run_root,
            max_wait_s=float(plan["max_wait_s"]),
            resume=False,
        ),
        cwd=REPO_ROOT,
        env=env,
        stdout_log=stdout,
        stderr_log=stderr,
    )
    returncode = process.wait()
    if returncode != 0:
        raise SHARED.AcceptanceError(
            f"{scenario} candidate batch failed rc={returncode}; "
            f"stderr={stderr}"
        )
    summary = _parse_json_output(stdout)
    return _validate_scenario(
        scenario=scenario,
        run_root=run_root,
        summary=summary,
        capture_root=scenario_root / "captures",
        evidence_root=scenario_root / "evidence",
    )


def _run_committed_cut(
    plan: Mapping[str, Any],
    *,
    scenario: str,
) -> dict[str, Any]:
    config = Path(plan["configs"][scenario])
    scenario_root = config.parent
    run_root = scenario_root / "run"
    stdout = scenario_root / "cut.stdout.log"
    stderr = scenario_root / "cut.stderr.log"
    env = _scenario_environment(plan, scenario)
    command = _command(
        config=config,
        run_root=run_root,
        max_wait_s=float(plan["max_wait_s"]),
        resume=False,
    )
    process = _launch(
        command=command,
        cwd=REPO_ROOT,
        env=env,
        stdout_log=stdout,
        stderr_log=stderr,
    )
    journal = run_root / ".framework" / "journal.sqlite"
    deadline = time.monotonic() + float(plan["max_wait_s"])
    while time.monotonic() < deadline:
        if _committed(journal):
            process.send_signal(signal.SIGKILL)
            process.wait()
            break
        if process.poll() is not None:
            raise SHARED.AcceptanceError(
                "candidate batch exited before the committed-round cut; "
                f"rc={process.returncode} stderr={stderr}"
            )
        time.sleep(0.002)
    else:
        process.kill()
        process.wait()
        raise SHARED.AcceptanceError(
            "timed out waiting for collection/commit before the cut"
        )

    resume_stdout = scenario_root / "resume.stdout.log"
    resume_stderr = scenario_root / "resume.stderr.log"
    resumed = _launch(
        command=_command(
            config=config,
            run_root=run_root,
            max_wait_s=float(plan["max_wait_s"]),
            resume=True,
        ),
        cwd=REPO_ROOT,
        env=env,
        stdout_log=resume_stdout,
        stderr_log=resume_stderr,
    )
    returncode = resumed.wait()
    if returncode != 0:
        raise SHARED.AcceptanceError(
            f"committed-cut resume failed rc={returncode}; "
            f"stderr={resume_stderr}"
        )
    summary = _parse_json_output(resume_stdout)
    result = _validate_scenario(
        scenario=scenario,
        run_root=run_root,
        summary=summary,
        capture_root=scenario_root / "captures",
        evidence_root=scenario_root / "evidence",
    )
    result["cut"] = {
        "signal": "SIGKILL",
        "boundary": "after collection/commit checkpoint",
        "parent_pid": process.pid,
        "parent_returncode": process.returncode,
        "resume_returncode": returncode,
    }
    return result


def _canonicalize_cif_timestamp(path: Path) -> tuple[str, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    canonical = []
    timestamps = []
    for line in lines:
        body = line[:-1] if line.endswith("\n") else line
        newline = "\n" if line.endswith("\n") else ""
        match = _MODEL_GROUP_TIMESTAMP.match(body)
        if match is None:
            canonical.append(line)
            continue
        timestamps.append(match.group("timestamp"))
        canonical.append(
            match.group("head")
            + "<TIMESTAMP>"
            + match.group("tail")
            + newline
        )
    return "".join(canonical), timestamps


def _compare_cif_pair(left_path: Path, right_path: Path) -> dict[str, Any]:
    """Reuse the established AF3 rule: only model-group timestamps may differ."""

    left_lines = left_path.read_text(encoding="utf-8").splitlines(keepends=True)
    right_lines = right_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(left_lines) != len(right_lines):
        raise SHARED.AcceptanceError(
            f"CIF line count differs: {left_path} vs {right_path}"
        )
    differences = []
    for line_number, (left_line, right_line) in enumerate(
        zip(left_lines, right_lines),
        1,
    ):
        if left_line == right_line:
            continue
        left_body = left_line[:-1] if left_line.endswith("\n") else left_line
        right_body = right_line[:-1] if right_line.endswith("\n") else right_line
        left_match = _MODEL_GROUP_TIMESTAMP.match(left_body)
        right_match = _MODEL_GROUP_TIMESTAMP.match(right_body)
        if left_match is None or right_match is None:
            raise SHARED.AcceptanceError(
                f"non-timestamp CIF difference: {left_path.name}:{line_number}"
            )
        left_canonical = (
            left_match.group("head")
            + "<TIMESTAMP>"
            + left_match.group("tail")
            + ("\n" if left_line.endswith("\n") else "")
        )
        right_canonical = (
            right_match.group("head")
            + "<TIMESTAMP>"
            + right_match.group("tail")
            + ("\n" if right_line.endswith("\n") else "")
        )
        if left_canonical != right_canonical:
            raise SHARED.AcceptanceError(
                "model_group_name differs outside the timestamp: "
                f"{left_path.name}:{line_number}"
            )
        differences.append(
            {
                "line": line_number,
                "left_timestamp": left_match.group("timestamp"),
                "right_timestamp": right_match.group("timestamp"),
            }
        )
    left_canonical, left_timestamps = _canonicalize_cif_timestamp(left_path)
    right_canonical, right_timestamps = _canonicalize_cif_timestamp(right_path)
    if left_canonical != right_canonical:
        raise SHARED.AcceptanceError(
            f"CIF content differs beyond model_group_name: {left_path.name}"
        )
    if len(left_timestamps) != len(right_timestamps):
        raise SHARED.AcceptanceError(
            f"model_group_name timestamp count differs: {left_path.name}"
        )
    return {
        "left_sha256": SHARED._sha256(left_path),
        "right_sha256": SHARED._sha256(right_path),
        "canonical_sha256": hashlib.sha256(
            left_canonical.encode("utf-8")
        ).hexdigest(),
        "timestamp_differences": differences,
        "all_other_cif_data_exact": True,
        "atom_ids_and_coordinates_exact": True,
    }


def _compare_child_evidence(
    *,
    left_root: Path,
    right_root: Path,
) -> dict[str, Any]:
    score_arrays = SHARED._compare_arrays(
        left_root / "standard_scores.npz",
        right_root / "standard_scores.npz",
    )
    rng_arrays = SHARED._compare_arrays(
        left_root / "rng_trace.npz",
        right_root / "rng_trace.npz",
    )
    coordinates = SHARED._compare_arrays(
        left_root / "structure_coordinates.npz",
        right_root / "structure_coordinates.npz",
    )
    score_metadata = SHARED._scientific_equal(
        SHARED._without_path_values(
            SHARED._load_json(left_root / "standard_scores.json")
        ),
        SHARED._without_path_values(
            SHARED._load_json(right_root / "standard_scores.json")
        ),
    )
    rng_metadata = SHARED._scientific_equal(
        SHARED._load_json(left_root / "rng_trace.json"),
        SHARED._load_json(right_root / "rng_trace.json"),
    )
    left_structures = SHARED._load_json(
        left_root / "structure_coordinates.json"
    )
    right_structures = SHARED._load_json(
        right_root / "structure_coordinates.json"
    )
    coordinate_identity = SHARED._scientific_equal(
        SHARED._coordinate_identity_projection(left_structures),
        SHARED._coordinate_identity_projection(right_structures),
    )
    if len(left_structures) != len(right_structures):
        raise SHARED.AcceptanceError("structure evidence counts differ")
    cif_comparisons = []
    for left_row, right_row in zip(left_structures, right_structures):
        cif_comparisons.append(
            _compare_cif_pair(
                Path(left_row["snapshot_path"]),
                Path(right_row["snapshot_path"]),
            )
        )
    passed = (
        score_arrays["equal"]
        and rng_arrays["equal"]
        and coordinates["equal"]
        and score_metadata
        and rng_metadata
        and coordinate_identity
    )
    if not passed:
        raise SHARED.AcceptanceError(
            "candidate evidence differs beyond the permitted AF3 CIF timestamps"
        )
    return {
        "status": "PASS",
        "score_arrays": score_arrays,
        "rng_arrays": rng_arrays,
        "structure_coordinates": coordinates,
        "score_metadata_except_artifact_paths": score_metadata,
        "rng_metadata": rng_metadata,
        "structure_identity_and_atoms": coordinate_identity,
        "cif_pairs": cif_comparisons,
    }


def _collection_projection(
    collections: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "collection_index": row.get("collection_index"),
            "trigger": row.get("trigger"),
            "selected_job_id": row.get("selected_job_id"),
            "selected_score": row.get("selected_score"),
            "completed": row.get("completed"),
            "failed": row.get("failed"),
            "failed_jobs": [
                {
                    "job_id": item.get("job_id"),
                    "error": item.get("error"),
                }
                for item in row.get("failed_jobs", [])
            ],
            "carried_forward_job_ids": row.get("carried_forward_jobs"),
        }
        for row in collections
    ]


def _compare_scenarios(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_run = Path(left["run_root"])
    right_run = Path(right["run_root"])
    children = {}
    for job_id in sorted(EXPECTED_CHILD_SEEDS):
        left_receipt = _read_receipt(left_run, job_id)
        right_receipt = _read_receipt(right_run, job_id)
        if (
            left_receipt.actual_seed != right_receipt.actual_seed
            or left_receipt.scientific_config != right_receipt.scientific_config
            or left_receipt.source_manifest != right_receipt.source_manifest
            or left_receipt.input_assets_manifest
            != right_receipt.input_assets_manifest
            or left_receipt.result is None
            or right_receipt.result is None
            or left_receipt.result.score != right_receipt.result.score
            or dict(left_receipt.result.payload.sequences)
            != dict(right_receipt.result.payload.sequences)
        ):
            raise SHARED.AcceptanceError(
                f"{job_id} receipt identity changed across scenarios"
            )
        children[job_id] = {
            "receipt_identity": "PASS",
            "evidence": _compare_child_evidence(
                left_root=Path(left["capture_evidence"][job_id]["evidence"]),
                right_root=Path(right["capture_evidence"][job_id]["evidence"]),
            ),
        }
    selection = {
        "selected_candidate_ids": (
            left.get("selected_candidate_ids") == right.get("selected_candidate_ids")
        ),
        "collections": (
            _collection_projection(left.get("collections", []))
            == _collection_projection(right.get("collections", []))
        ),
    }
    if not all(selection.values()):
        raise SHARED.AcceptanceError(
            "candidate selection or collection rows changed across scenarios"
        )
    return {
        "status": "PASS",
        "children": children,
        "selection": selection,
        "only_allowed_difference": "AF3 CIF model_group_name timestamps",
    }


def _execute(plan: Mapping[str, Any], out_root: Path) -> dict[str, Any]:
    fingerprint = _source_fingerprint(_source_manifest())
    if fingerprint != plan["source_fingerprint"]:
        raise SHARED.AcceptanceError(
            "Astrolabe source changed between candidate recovery prepare and execute"
        )
    nvidia = SHARED._nvidia_manifest()
    uninterrupted = _run_uninterrupted(plan, scenario="uninterrupted")
    committed_cut = _run_committed_cut(plan, scenario="committed_cut")
    total_requests = (
        uninterrupted["request_count"] + committed_cut["request_count"]
    )
    if total_requests > TOTAL_REQUEST_BUDGET:
        raise SHARED.AcceptanceError(
            f"candidate recovery acceptance used {total_requests} requests, "
            f"above hard cap {TOTAL_REQUEST_BUDGET}"
        )
    comparison = _compare_scenarios(uninterrupted, committed_cut)
    result = {
        "schema": SCHEMA,
        "status": "passed",
        "source_fingerprint": fingerprint,
        "nvidia": nvidia,
        "scenarios": {
            "uninterrupted": uninterrupted,
            "committed_cut": committed_cut,
        },
        "request_accounting": {
            "uninterrupted": uninterrupted["request_count"],
            "committed_cut_including_resume": committed_cut["request_count"],
            "total": total_requests,
            "hard_cap": TOTAL_REQUEST_BUDGET,
        },
        "comparison": comparison,
    }
    SHARED._atomic_json(out_root / "result.json", result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=REPO_ROOT / "config" / "container_af3_smoke.yaml",
    )
    parser.add_argument(
        "--max-wait-s",
        type=float,
        default=VALIDATION_MAX_WAIT_S,
        help="Fixed validation collection window; production default remains 30s.",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_root = args.out_root.expanduser().resolve()
    profile = args.profile.expanduser().resolve()
    if args.execute and (out_root / "plan.json").is_file():
        plan = _load_prepared_plan(
            out_root=out_root,
            profile=profile,
            max_wait_s=float(args.max_wait_s),
        )
    else:
        plan = _candidate_plan(
            out_root=out_root,
            profile=profile,
            max_wait_s=float(args.max_wait_s),
        )
    if args.execute:
        _execute(plan, out_root)
    else:
        print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
