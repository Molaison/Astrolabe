from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "validation" / "posthoc_parallel_final.py"
)


def _load_parallel():
    name = "posthoc_parallel_final_test_module"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepare(tmp_path: Path):
    parallel = _load_parallel()
    prepared = parallel.prepare_acceptance(
        tmp_path / "parallel",
        repo_root=REPO_ROOT,
        model_dir=Path("/models/alphafold3"),
    )
    return parallel, prepared


def test_prepare_freezes_two_scientific_profiles_and_four_request_budget(
    tmp_path,
):
    parallel, prepared = _prepare(tmp_path)
    manifest = Path(prepared["inputs"]["manifest"])
    records = parallel._read_jsonl(manifest)

    assert prepared["request_contract"] == {
        "native_baseline": 2,
        "public_actual_workers2": 2,
        "public_parent_kill_recovery": 0,
        "public_parallel_cache": 0,
        "public_serial_cache": 0,
        "maximum_new_model_requests": 4,
    }
    assert [profile["num_recycles"] for profile in prepared["profiles"]] == [
        1,
        2,
    ]
    assert [profile["seed"] for profile in prepared["profiles"]] == [17, 17]
    assert [profile["num_samples"] for profile in prepared["profiles"]] == [
        1,
        1,
    ]
    assert [record["seqs"] for record in records] == [
        {"A": parallel.SEQUENCE},
        {"A": parallel.SEQUENCE},
    ]
    assert len([record["src_run_dir"] for record in records]) == 2
    assert len(set(record["src_run_dir"] for record in records)) == 2
    assert prepared["commands"]["public_actual_workers2"][
        prepared["commands"]["public_actual_workers2"].index("--workers") + 1
    ] == "2"
    assert prepared["commands"]["public_serial_cache_workers1"][
        prepared["commands"]["public_serial_cache_workers1"].index("--workers")
        + 1
    ] == "1"
    assert prepared["control_contract"]["groups"] == 2
    assert prepared["control_contract"]["resource_groups"] == 2
    assert prepared["control_contract"]["shard_child_workers"] == 1
    assert prepared["control_contract"]["slot_worker_execution"] == (
        "synchronous"
    )
    interface = parallel._fixedlist_interface_evidence(prepared)
    assert interface["status"] == "PASS"
    assert interface["synchronous_slot_worker"] is True


def test_verify_prepared_rejects_manifest_drift(tmp_path):
    parallel, prepared = _prepare(tmp_path)
    out_root = Path(prepared["out_root"])
    assert parallel._verify_prepared(out_root)["profiles"] == prepared["profiles"]

    manifest = Path(prepared["inputs"]["manifest"])
    records = parallel._read_jsonl(manifest)
    records[0]["seqs"]["A"] = "A" * 20
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    try:
        parallel._verify_prepared(out_root)
    except parallel.ParallelAcceptanceError as error:
        assert "frozen input changed" in str(error)
    else:
        raise AssertionError("manifest drift was accepted")


def test_empty_npz_comparison_fails_closed(tmp_path):
    parallel = _load_parallel()
    harness = parallel._load_harness()
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    np.savez(left)
    np.savez(right)
    result = harness.compare_npz(left, right)
    assert result["equal"] is False
    assert result["array_count"] == 0
    assert result["failures"] == [
        {"key": None, "reason": "empty-array-archive"}
    ]


def test_parent_kill_recovery_precedes_caches_and_uses_public_command(
    tmp_path,
    monkeypatch,
):
    parallel = _load_parallel()
    out_root = tmp_path / "parallel"
    commands = {
        "public_actual_workers2": ["public", "actual"],
        "public_parallel_cache_workers2": ["parallel", "cache"],
        "public_serial_cache_workers1": ["serial", "cache"],
    }
    prepared = {"commands": commands}
    artifact_paid = {
        "fingerprint": "paid",
        "files": {"receipt.json": "a" * 64},
    }
    snapshots = iter(
        [
            dict(artifact_paid),
            dict(artifact_paid),
            dict(artifact_paid),
        ]
    )
    calls = []

    def _cache(**kwargs):
        calls.append(kwargs)
        return {
            "label": kwargs["label"],
            "requests": {"observed_model_requests": 0},
        }

    class _Harness:
        @staticmethod
        def _tree_snapshot(_root):
            return next(snapshots)

    monkeypatch.setattr(parallel, "_invoke_cache", _cache)
    monkeypatch.setattr(parallel, "_load_harness", lambda: _Harness)

    result = parallel._run_public_recovery_and_caches(
        out_root=out_root,
        prepared=prepared,
        artifact_paid=artifact_paid,
        timeout_s=123,
    )

    assert [call["label"] for call in calls] == [
        "public_parent_kill_recovery_workers2",
        "public_parallel_cache_workers2",
        "public_serial_cache_workers1",
    ]
    assert calls[0]["command"] == commands["public_actual_workers2"]
    assert calls[0]["timeout_s"] == 123
    assert result["artifact_fingerprints"]["unchanged"] is True
    assert result["public_parent_kill_recovery"]["label"] == (
        "public_parent_kill_recovery_workers2"
    )


def test_global_request_claims_are_unique_across_processes(tmp_path):
    parallel = _load_parallel()
    harness = parallel._load_harness()
    capture = tmp_path / "capture"
    harness._install_capture(capture)
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
            "ASTROLABE_POSTHOC_ACCEPTANCE_STAGE": "cpu-global-ledger",
            "ASTROLABE_POSTHOC_ACCEPTANCE_MAX_REQUESTS": "2",
            "ASTROLABE_POSTHOC_ACCEPTANCE_GLOBAL_LEDGER": "1",
        }
    )
    code = (
        "import sitecustomize\n"
        "print(sitecustomize._global_claim('request', maximum=2))\n"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    outputs = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        outputs.append(int(stdout.strip()))
    assert sorted(outputs) == [1, 2]

    central = tmp_path / "central-ledger"
    total_code = (
        "import sys\n"
        "from pathlib import Path\n"
        "import sitecustomize\n"
        "print(sitecustomize._global_claim("
        "'request_total', maximum=5, root=Path(sys.argv[1])))\n"
    )
    totals = []
    for _ in range(5):
        completed = subprocess.run(
            [sys.executable, "-c", total_code, str(central)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        totals.append(int(completed.stdout.strip()))
    assert totals == [1, 2, 3, 4, 5]


def test_real_generated_entrypoint_preflights_are_model_free(tmp_path):
    parallel, prepared = _prepare(tmp_path)
    out_root = Path(prepared["out_root"])
    results = parallel._run_entrypoint_preflights(
        out_root=out_root,
        prepared=prepared,
        timeout_s=120,
    )

    assert set(results) == {"rec1", "rec2"}
    for profile_name, rec in (("rec1", 1), ("rec2", 2)):
        evidence = results[profile_name]["evidence"]
        assert evidence["status"] == "PASS"
        assert evidence["model_execution"] is False
        assert evidence["gpu_model_calls"] == 0
        assert evidence["inner_guard_reached"] is True
        assert evidence["request_counts"] == {
            "rpc_delivered": 1,
            "request_reservations": 1,
            "inner_run_inference_invocations": 1,
            "inner_run_inference_completed": 0,
            "entrypoint_completed": 0,
            "failed_or_denied_model_requests": 1,
        }
        assert evidence["rng"]["keys"] == ["rng_key"]
        assert evidence["rng"]["dtype_shape"]["rng_key"] == {
            "dtype": "<u4",
            "shape": [2],
        }
        assert evidence["resource_owner"]["in_process_runner"] is True
        assert evidence["resource_owner"]["num_recycles"] == rec


def test_fixedlist_cut_blocks_before_collect_all_without_model(
    tmp_path,
):
    parallel = _load_parallel()
    harness = parallel._load_harness()
    root = tmp_path / "cut"
    capture = root / "capture"
    caller_root = root / "caller/runs"
    ready_path = root / "cut-ready.json"
    harness._install_capture(capture)
    receipts = parallel._expected_receipt_paths(caller_root)
    coordinator = root / "fixedlist_coordinator.py"
    coordinator.write_text(
        "\n".join(
            [
                "import subprocess",
                "import sys",
                "from pathlib import Path",
                "from astrolabe.design.fixed_list import FixedTask",
                "from astrolabe.posthoc import refold",
                "root = Path(sys.argv[1])",
                "receipts = {",
                "    'shard_0001': root / 'caller/runs/posthoc_parallel_final/single20_seed17_rec1/.framework/refold/af3/step_000_A/receipt.json',",
                "    'shard_0002': root / 'caller/runs/posthoc_parallel_final/single20_seed17_rec2/.framework/refold/af3/step_000_A/receipt.json',",
                "}",
                "summaries = {",
                "    'shard_0001': root / 'caller/runs/_parallel_workers/run_fixture/worker_1.summary.json',",
                "    'shard_0002': root / 'caller/runs/_parallel_workers/run_fixture/worker_2.summary.json',",
                "}",
                "child = r'''",
                "# from astrolabe.posthoc.refold import run_refold",
                "import json, os, sys",
                "from pathlib import Path",
                "receipt = Path(sys.argv[1])",
                "summary = Path(sys.argv[2])",
                "receipt.parent.mkdir(parents=True, exist_ok=True)",
                "summary.parent.mkdir(parents=True, exist_ok=True)",
                "receipt.write_text(json.dumps({'schema':'astrolabe.posthoc-refold.receipt/v1','status':'completed','backend':'af3','origin':'inference','identity_sha256':'a'*64,'outputs':{'fingerprint':'b'*64},'step':0,'oligo':'A'}), encoding='utf-8')",
                "summary.write_text(json.dumps({'workers_requested':1,'workers':1,'processed':1,'skipped':0,'failed':0,'ran_af3':1,'receipt_cache_hits_af3':0,'groups':1,'resource_groups':1}), encoding='utf-8')",
                "'''",
                "def _run_shard(job):",
                "    subprocess.run([sys.executable, '-c', child, str(receipts[job.job_id]), str(summaries[job.job_id])], check=True)",
                "    return 0.0",
                "result = refold.run_fixed_task_list((FixedTask('shard_0001'), FixedTask('shard_0002')), _run_shard, max_in_flight=2)",
                "print('COLLECT_ALL_RETURNED')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
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
            "ASTROLABE_POSTHOC_ACCEPTANCE_STAGE": "cpu-cut",
            "ASTROLABE_POSTHOC_ACCEPTANCE_MAX_REQUESTS": "0",
            "ASTROLABE_POSTHOC_PARALLEL_CUT": "1",
            "ASTROLABE_POSTHOC_PARALLEL_CUT_EXPECTED": "2",
            "ASTROLABE_POSTHOC_PARALLEL_CUT_ROOT": str(caller_root),
            "ASTROLABE_POSTHOC_PARALLEL_CUT_READY": str(ready_path),
        }
    )
    stdout_path = root / "coordinator.stdout"
    stderr_path = root / "coordinator.stderr"
    with stdout_path.open("w", encoding="utf-8") as stdout, (
        stderr_path.open("w", encoding="utf-8")
    ) as stderr:
        parent = subprocess.Popen(
            [sys.executable, str(coordinator), str(root)],
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    identity = parallel._process_identity(parent.pid)
    deadline = time.monotonic() + 30
    while not ready_path.exists():
        assert parent.poll() is None
        assert time.monotonic() < deadline
        time.sleep(0.05)
    ready = parallel._validate_cut_ready(
        ready_path=ready_path,
        parent_pid=parent.pid,
        caller_root=caller_root,
    )
    assert ready["boundary"] == (
        "refold_run_fixed_task_list_after_child_wait_"
        "and_receipt_commit_before_result_consume"
    )
    assert ready["outer_worker_name"] == "_run_shard"
    assert ready["max_in_flight"] == 2
    assert ready["resume_contract"] == (
        "per_slot_committed_receipts_with_no_outer_journal"
    )
    assert "COLLECT_ALL_RETURNED" not in stdout_path.read_text(
        encoding="utf-8"
    )
    stopped = parallel._wait_for_state(
        parent.pid,
        states={"T", "t"},
        timeout_s=10,
    )
    assert stopped["starttime_ticks"] == identity["starttime_ticks"]
    assert stopped["boot_id"] == identity["boot_id"]
    os.kill(parent.pid, signal.SIGKILL)
    assert parent.wait(timeout=30) == -signal.SIGKILL
    assert "COLLECT_ALL_RETURNED" not in stdout_path.read_text(
        encoding="utf-8"
    )
