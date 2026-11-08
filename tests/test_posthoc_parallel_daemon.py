from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "validation" / "posthoc_parallel_daemon.py"
)
BASELINE_ATTEMPT = Path(
    os.environ.get(
        "ASTROLABE_POSTHOC_BASELINE_ATTEMPT",
        str(
            REPO_ROOT
            / "outputs"
            / "unified-gpu-acceptance"
            / "posthoc_parallel_final_v2"
            / "attempt_001"
        ),
    )
)


def _load_daemon():
    name = "posthoc_parallel_daemon_test_module"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepare(tmp_path: Path):
    parallel = _load_daemon()
    prepared = parallel.prepare_acceptance(
        tmp_path / "parallel-daemon",
        repo_root=REPO_ROOT,
        model_dir=Path("/models/alphafold3"),
        baseline_attempt=BASELINE_ATTEMPT,
    )
    return parallel, prepared


def test_prepare_reuses_two_baselines_and_only_changes_transport(
    tmp_path,
):
    parallel, prepared = _prepare(tmp_path)

    assert prepared["request_contract"] == {
        "native_baseline_reused": 2,
        "native_baseline_new": 0,
        "public_daemon_workers2": 2,
        "public_parent_kill_recovery": 0,
        "public_parallel_cache": 0,
        "public_serial_cache": 0,
        "maximum_new_model_requests": 4,
    }
    assert "native_baseline" not in prepared["commands"]
    assert prepared["baseline_provenance"]["numeric_source_unchanged"] is True

    for profile in parallel.PROFILES:
        record = prepared["baseline_reuse"][profile.name]
        assert record["profile"] == profile.name
        assert record["requests"]["observed_model_requests"] == 1
        assert record["scientific_config_equal_after_transport_normalization"]
        assert record["transport_differences"] == [
            {
                "path": (
                    "evaluator.alphafold3.execution"
                ),
                "baseline": "in_process",
                "daemon": "daemon",
            },
            {
                "path": (
                    "evaluator.evaluator_terms."
                    "Alphafold3.execution"
                ),
                "baseline": "in_process",
                "daemon": "daemon",
            },
        ]
        for value in record["evidence_hashes"].values():
            assert Path(value["path"]).is_file()
            assert len(value["sha256"]) == 64


def test_generated_daemon_sitecustomize_runs_in_fresh_process_without_model(
    tmp_path,
):
    parallel = _load_daemon()
    capture = tmp_path / "capture"
    output = tmp_path / "output"
    parallel._install_capture(capture)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(capture), str(REPO_ROOT)]
            ),
            "PYTHONNOUSERSITE": "1",
            "JAX_PLATFORMS": "cpu",
            "ASTROLABE_POSTHOC_ACCEPTANCE_CAPTURE": "1",
            "ASTROLABE_POSTHOC_ACCEPTANCE_CAPTURE_DIR": str(capture),
            "ASTROLABE_POSTHOC_ACCEPTANCE_STAGE": "daemon-preflight-test",
            "ASTROLABE_POSTHOC_ACCEPTANCE_MAX_REQUESTS": "0",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "_daemon-preflight-worker",
            "--baseline-capture",
            str(
                BASELINE_ATTEMPT
                / "evidence/requests/native_baseline_rec1"
            ),
            "--out-root",
            str(output),
        ],
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(
        (output / "daemon_preflight.json").read_text()
    )
    assert report["status"] == "PASS"
    assert report["transport_shape_only"] is True
    assert report["model_execution"] is False
    assert report["new_model_requests"] == 0
    events = parallel._events(capture)
    kinds = [event["event"] for event in events]
    assert kinds.count("model_daemon_rpc_started") == 1
    assert kinds.count("model_daemon_rpc_completed") == 1
    assert not any(event.startswith("model_inner_") for event in kinds)


def test_daemon_transport_rejects_client_daemon_identity_confusion(
    tmp_path,
):
    parallel = _load_daemon()
    capture = tmp_path / "capture"
    capture.mkdir()
    lifecycle = tmp_path / "lifecycle.json"
    lifecycle.write_text(
        json.dumps(
            {
                "status": "stopped",
                "pid": 200,
                "parent_pid": 100,
                "exit_code": 0,
            }
        )
    )
    lifecycle_2 = tmp_path / "lifecycle_2.json"
    lifecycle_2.write_text(
        json.dumps(
            {
                "status": "stopped",
                "pid": 201,
                "parent_pid": 101,
                "exit_code": 0,
            }
        )
    )
    log = tmp_path / "owner.jsonl"
    log.write_text("")
    log_2 = tmp_path / "owner_2.jsonl"
    log_2.write_text("")
    owner = {"ppid": 100, "num_recycles": 1}
    events = [
        {
            "event": "model_daemon_rpc_started",
            "client_pid": 100,
            "daemon_pid": 100,
            "daemon_model_load_count": 1,
            "daemon_request_count": 1,
            "daemon_completed_count": 1,
            "daemon_failed_count": 0,
        },
        {
            "event": "model_daemon_rpc_started",
            "client_pid": 101,
            "daemon_pid": 201,
            "daemon_model_load_count": 1,
            "daemon_request_count": 1,
            "daemon_completed_count": 1,
            "daemon_failed_count": 0,
        },
        {
            "event": "model_daemon_rpc_completed",
            "client_pid": 100,
            "client_ppid": 50,
            "daemon_pid": 100,
            "daemon_parent_pid": 100,
            "transport": "daemon",
            "in_process_runner": False,
            "daemon_model_load_count": 1,
            "daemon_request_count": 2,
            "daemon_completed_count": 2,
            "daemon_failed_count": 0,
            "daemon_lifecycle_path": str(lifecycle),
            "daemon_log_path": str(log),
            "num_recycles": 1,
            "num_samples": 1,
            "time_ns": 1,
        },
        {
            "event": "model_daemon_rpc_completed",
            "client_pid": 101,
            "client_ppid": 50,
            "daemon_pid": 201,
            "daemon_parent_pid": 101,
            "transport": "daemon",
            "in_process_runner": False,
            "daemon_model_load_count": 1,
            "daemon_request_count": 2,
            "daemon_completed_count": 2,
            "daemon_failed_count": 0,
            "daemon_lifecycle_path": str(lifecycle_2),
            "daemon_log_path": str(log_2),
            "num_recycles": 2,
            "num_samples": 1,
            "time_ns": 2,
        },
        {
            "event": "model_inner_invocation_started",
            "pid": 100,
            "resource_owner": owner,
            "time_ns": 1,
        },
        {
            "event": "model_inner_invocation_completed",
            "pid": 100,
            "resource_owner": owner,
            "request_id": 1,
            "time_ns": 2,
        },
    ]
    (capture / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    try:
        parallel._daemon_transport_evidence(capture)
    except parallel.ParallelAcceptanceError as error:
        assert "client PID" in str(error)
    else:
        raise AssertionError("client/daemon PID collision was accepted")


def test_daemon_transport_requires_two_clean_owners_and_inner_matches(
    tmp_path,
):
    parallel = _load_daemon()
    capture = tmp_path / "capture"
    capture.mkdir()
    events = []
    for rec, client_pid, daemon_pid, start_ns in (
        (1, 100, 200, 10),
        (2, 101, 201, 11),
    ):
        lifecycle = tmp_path / f"lifecycle_{rec}.json"
        lifecycle.write_text(
            json.dumps(
                {
                    "status": "stopped",
                    "pid": daemon_pid,
                    "parent_pid": client_pid,
                    "exit_code": 0,
                }
            )
        )
        log = tmp_path / f"owner_{rec}.jsonl"
        log.write_text("")
        owner = {
            "pid": daemon_pid,
            "ppid": client_pid,
            "num_recycles": rec,
            "device": {"platform": "gpu", "kind": "fixture-gpu", "id": 0},
        }
        events.extend(
            [
                {
                    "event": "model_daemon_rpc_started",
                    "client_pid": client_pid,
                    "daemon_pid": daemon_pid,
                    "daemon_model_load_count": 1,
                    "daemon_request_count": 1,
                    "daemon_completed_count": 1,
                    "daemon_failed_count": 0,
                },
                {
                    "event": "model_daemon_rpc_completed",
                    "client_pid": client_pid,
                    "client_ppid": 50,
                    "daemon_pid": daemon_pid,
                    "daemon_parent_pid": client_pid,
                    "transport": "daemon",
                    "in_process_runner": False,
                    "daemon_model_load_count": 1,
                    "daemon_request_count": 2,
                    "daemon_completed_count": 2,
                    "daemon_failed_count": 0,
                    "daemon_identity": str(rec) * 64,
                    "daemon_socket": str(tmp_path / f"{rec}.sock"),
                    "daemon_lifecycle_path": str(lifecycle),
                    "daemon_log_path": str(log),
                    "num_recycles": rec,
                    "num_samples": 1,
                    "model_dir": "/models/alphafold3",
                    "time_ns": start_ns + 2,
                },
                {
                    "event": "model_inner_invocation_started",
                    "pid": daemon_pid,
                    "resource_owner": owner,
                    "time_ns": start_ns,
                },
                {
                    "event": "model_inner_invocation_completed",
                    "pid": daemon_pid,
                    "resource_owner": owner,
                    "request_id": rec,
                    "time_ns": start_ns + 100,
                },
            ]
        )
    (capture / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )

    evidence = parallel._daemon_transport_evidence(capture)

    assert evidence["transport"] == "daemon"
    assert evidence["request_count"] == 2
    assert evidence["distinct_clients"] == 2
    assert evidence["distinct_daemons"] == 2
    assert evidence["overlap_ns"] == 99
    assert [
        record["start_status_counts"]["requests"]
        for record in evidence["records"]
    ] == [1, 1]
    assert [
        record["end_status_counts"]["requests"]
        for record in evidence["records"]
    ] == [2, 2]
    assert [record["daemon_exit_code"] for record in evidence["records"]] == [
        0,
        0,
    ]


def test_input_snapshot_comparison_normalizes_only_empty_strides(tmp_path):
    parallel = _load_daemon()
    empty = {
        "kind": "numeric_array",
        "dtype": "<i8",
        "shape": [0],
        "size": 0,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "strides": [8],
    }
    snapshot = {
        "schema": "astrolabe.posthoc-refold.featurised-input-snapshot/v1",
        "scope": {"unsupported_objects": "fail closed"},
        "counts": {"array_count": 1},
        "tree": {"items": {"empty": empty}},
    }
    public_path = tmp_path / "public.json"
    baseline_path = tmp_path / "baseline.json"
    public_path.write_text(json.dumps(snapshot))
    baseline_snapshot = json.loads(json.dumps(snapshot))
    baseline_snapshot["tree"]["items"]["empty"]["strides"] = [0]
    baseline_path.write_text(json.dumps(baseline_snapshot))
    completion = lambda path: {"input_snapshot_path": str(path)}

    comparison = parallel._input_snapshot_comparison(
        completion(public_path),
        completion(baseline_path),
    )

    assert comparison["equal"] is True
    assert comparison["raw_tree_digests_equal"] is False
    assert len(comparison["normalization"]["public_changed_paths"]) == 1
    assert comparison["normalization"]["baseline_changed_paths"] == []
    assert (
        comparison["public"]["sha256"]
        == comparison["baseline"]["sha256"]
    )

    baseline_snapshot["tree"]["items"]["empty"]["sha256"] = "not-empty"
    baseline_path.write_text(json.dumps(baseline_snapshot))
    comparison = parallel._input_snapshot_comparison(
        completion(public_path),
        completion(baseline_path),
    )
    assert comparison["equal"] is False


def test_featurised_input_digest_canonicalizes_empty_ndarray_stride(
    tmp_path,
):
    parallel = _load_daemon()
    capture = tmp_path / "capture"
    parallel._install_capture(capture)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(capture),
            "PYTHONNOUSERSITE": "1",
        }
    )
    code = """
import json
import numpy as np
import sitecustomize

empty = np.empty(0, dtype=np.int64)
transport = np.lib.stride_tricks.as_strided(
    empty,
    shape=(0,),
    strides=(np.dtype(np.int64).itemsize,),
)
baseline_digest, baseline_snapshot = sitecustomize._digest_tree(empty)
transport_digest, transport_snapshot = sitecustomize._digest_tree(transport)
print(json.dumps({
    "baseline_strides": baseline_snapshot["tree"]["strides"],
    "transport_strides": transport_snapshot["tree"]["strides"],
    "baseline_sha256": baseline_digest["sha256"],
    "transport_sha256": transport_digest["sha256"],
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)

    assert result["baseline_strides"] == [0]
    assert result["transport_strides"] == [0]
    assert result["baseline_sha256"] == result["transport_sha256"]


def test_daemon_config_drift_fails_before_model_execution(tmp_path):
    parallel, prepared = _prepare(tmp_path)
    out_root = Path(prepared["out_root"])
    config = Path(prepared["inputs"]["configs"]["rec1"])
    config.write_text(
        config.read_text().replace(
            "num_recycles: 1",
            "num_recycles: 3",
            1,
        )
    )
    try:
        parallel._verify_prepared(out_root)
    except parallel.ParallelAcceptanceError as error:
        assert "frozen input changed" in str(error)
    else:
        raise AssertionError("daemon config drift was accepted")
