from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import json
import os
import pickle
from pathlib import Path
import signal
import subprocess
import sys
import time

from omegaconf import OmegaConf, open_dict
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "scripts" / "validation" / "isolated_gpu_acceptance.py"
CASE_FIXTURE = REPO_ROOT / "tests" / "data" / "protein_beam" / "02_fixture.yaml"


def _load_harness():
    name = "isolated_gpu_acceptance_harness_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_cif(path: Path, *, x: float = 1.25) -> None:
    path.write_text(
        "\n".join(
            [
                "data_test",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.label_atom_id",
                "_atom_site.label_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_seq_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                f"ATOM CA ALA A A 1 {x} 2.0 3.0",
                "#",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_native_run(
    root: Path,
    *,
    x: float = 1.25,
    score: float = 0.125,
    score_array_value: float = 0.75,
) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    output = root / "output"
    config = root / "config.yaml"
    structure = root / "model.cif"
    _write_cif(structure, x=x)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "seed": 10079,
                    "out_dir": str(output),
                    "raw_model_results_dir": str(output / "raw"),
                    "protein_beam": {
                        "max_evaluations": 2,
                        "af3_call_budget": {
                            "ledger": str(output / "af3_calls.json"),
                            "max_total": 2,
                            "max_per_case": 2,
                        },
                    },
                }
            )
        ),
        encoding="utf-8",
    )
    for candidate_id, sequence, candidate_score in (
        (0, "ACDE", score),
        (1, "ACDF", score + 0.25),
    ):
        candidate = output / "candidates" / f"{candidate_id:06d}"
        candidate.mkdir(parents=True)
        from astrolabe.evaluators.utils import StandardScores

        snapshot = candidate / "scores.pkl"
        scores = StandardScores(
            names=["AB"],
            plddt={
                "AB": [
                    np.asarray(
                        [score_array_value, score_array_value + candidate_id * 0.01],
                        dtype=np.float32,
                    )
                ]
            },
            complex_paths={"AB": structure},
            ptm={"AB": [0.5]},
            pae={
                "AB": [
                    np.asarray(
                        [[score_array_value, 0.25], [0.25, score_array_value]],
                        dtype=np.float32,
                    )
                ]
            },
        )
        with snapshot.open("wb") as handle:
            pickle.dump(scores, handle, protocol=5)
        (candidate / "result.json").write_text(
            json.dumps(
                {
                    "id": candidate_id,
                    "status": "done",
                    "sequence": sequence,
                    "score": candidate_score,
                    "metrics": {"interface_ipae": 0.25},
                    "geometry": {"root_binder_pose_ca_rmsd": 0.125},
                    "structure_path": str(structure),
                    "snapshot": str(snapshot),
                    "evaluation_origin": "fresh_af3",
                    "eligible": False,
                }
            ),
            encoding="utf-8",
        )
    (output / "af3_calls.json").write_text(
        json.dumps(
            {
                "contract": {"max_total": 2, "max_per_case": 2},
                "entries": {
                    "scope:0:1": {
                        "candidate_id": 0,
                        "attempt": 1,
                        "seed": 10079,
                        "status": "issued",
                    },
                    "scope:1:1": {
                        "candidate_id": 1,
                        "attempt": 1,
                        "seed": 10079,
                        "status": "issued",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    service = output / "services" / "owned"
    service.mkdir(parents=True)
    (service / "lifecycle.json").write_text(
        json.dumps({"started_at": 1.0, "status": "drained"}),
        encoding="utf-8",
    )
    events = [
        {"event": "service_ready", "pid": 123},
        {
            "event": "request_received",
            "request_id": 1,
            "identity": {"candidate_id": 0, "seed": 10079},
        },
        {
            "event": "gpu_start",
            "request_id": 1,
            "identity": {"candidate_id": 0, "seed": 10079},
        },
        {
            "event": "gpu_end",
            "request_id": 1,
            "identity": {"candidate_id": 0, "seed": 10079},
            "ok": True,
        },
        {
            "event": "request_received",
            "request_id": 2,
            "identity": {"candidate_id": 1, "seed": 10079},
        },
        {
            "event": "gpu_start",
            "request_id": 2,
            "identity": {"candidate_id": 1, "seed": 10079},
        },
        {
            "event": "gpu_end",
            "request_id": 2,
            "identity": {"candidate_id": 1, "seed": 10079},
            "ok": True,
        },
    ]
    (service / "owner.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return output, config


def test_budget_changes_only_native_budget_and_quota(tmp_path: Path) -> None:
    harness = _load_harness()
    from astrolabe.protein_beam import load_config

    original = load_config(CASE_FIXTURE)
    cfg = harness._bounded_config(CASE_FIXTURE, tmp_path / "output")

    assert cfg.seed == original.seed
    assert cfg.seqs == original.seqs
    assert cfg.mutator == original.mutator
    assert cfg.loss == original.loss
    assert cfg.stages == original.stages
    assert cfg.evaluator.alphafold3.num_samples == 1
    assert cfg.evaluator.alphafold3.num_recycles == 3
    assert cfg.protein_beam.max_evaluations == harness.EVALUATION_BUDGET
    assert cfg.protein_beam.max_generations is None
    assert cfg.protein_beam.af3_call_budget.max_total == 2
    assert cfg.protein_beam.af3_call_budget.max_per_case == 2


def test_full_run_gate_rejects_engineering_fixture(tmp_path: Path) -> None:
    harness = _load_harness()
    cfg = harness._bounded_config(CASE_FIXTURE, tmp_path / "output")
    reasons = harness._full_run_reasons(cfg, CASE_FIXTURE)
    assert any("case_input.structure" in reason for reason in reasons)
    assert any("homologies.B" in reason for reason in reasons)
    assert any("root calibration" in reason for reason in reasons)
    assert any("geometry gates" in reason for reason in reasons)
    assert not any("AF3 service settings are invalid" in reason for reason in reasons)


def test_full_run_gate_uses_shared_invalid_service_validation(tmp_path: Path) -> None:
    harness = _load_harness()
    cfg = harness._bounded_config(CASE_FIXTURE, tmp_path / "output")
    with open_dict(cfg.protein_beam):
        cfg.protein_beam.service = {"max_pending": 0}
    reasons = harness._full_run_reasons(cfg, CASE_FIXTURE)
    assert any(
        "AF3 service settings are invalid" in reason
        and "max_pending" in reason
        for reason in reasons
    )


def test_prepare_only_accepts_directory_and_does_not_open_an_owner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    config_dir = tmp_path / "case"
    config_dir.mkdir()
    (config_dir / "02_fixture.yaml").write_bytes(CASE_FIXTURE.read_bytes())
    out_root = tmp_path / "acceptance"

    def no_owner(*args, **kwargs):
        raise AssertionError("prepare-only opened an AF3 owner")

    monkeypatch.setattr(harness, "_coordinator_main", no_owner)
    assert harness.main(
        [
            "--out-root",
            str(out_root),
            "--case-config",
            str(config_dir),
            "--prepare-only",
        ]
    ) == 0
    plan = json.loads((out_root / "plan.json").read_text())
    assert plan["status"] == "prepared"
    assert plan["evaluation_budget_including_root"] == 2
    assert plan["af3_quota"] == {"max_total": 2, "max_per_case": 2}
    assert plan["runs"]["baseline"]["service_root"].endswith(
        "/runs/baseline-service"
    )
    assert plan["runs"]["recovery"]["service_root"].endswith(
        "/runs/recovery-service"
    )
    assert plan["runs"]["recovery"]["resume"] is True
    assert plan["runs"]["recovery"]["retry_interrupted"] is False
    assert plan["full_run_ready"] is False
    assert not (out_root / "runs").exists()


def test_actual_owned_service_layout_is_sibling_of_untouched_beam_output(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    from astrolabe.evaluators.af3_lifecycle import OwnedAF3
    from astrolabe.protein_beam import load_config

    prepared = harness._prepare_acceptance(
        CASE_FIXTURE,
        tmp_path / "acceptance",
    )
    assert prepared.baseline_service_root is not None
    owner = OwnedAF3(
        load_config(prepared.baseline_config),
        prepared.baseline_service_root,
        os.environ,
    )
    assert owner.directory.parent == prepared.baseline_service_root / "services"
    assert not (prepared.baseline_output / "services").exists()
    assert not prepared.baseline_output.exists()
    assert prepared.baseline_output.parent == prepared.baseline_service_root.parent


def test_resume_missing_terminal_stops_before_owned_model_start(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    import astrolabe.evaluators.af3_lifecycle as lifecycle

    output = tmp_path / "runs" / "recovery"
    service_root = tmp_path / "runs" / "recovery-service"
    output.mkdir(parents=True)
    (output / "beam_state.json").write_text(
        json.dumps(
            {
                "status": "interrupted",
                "records": [{"id": 0, "status": "running"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        lifecycle,
        "recover_owned_services",
        lambda root: None,
    )
    monkeypatch.setattr(
        lifecycle,
        "OwnedAF3",
        lambda *args, **kwargs: pytest.fail(
            "model owner started before interruption preflight"
        ),
    )
    arguments = type(
        "Arguments",
        (),
        {
            "config": CASE_FIXTURE,
            "output": output,
            "service_root": service_root,
            "resume": True,
            "source_fingerprint": harness._source_fingerprint(
                harness._source_manifest()
            ),
        },
    )()
    with pytest.raises(RuntimeError, match="lacks a terminal result"):
        harness._coordinator_main(arguments)


def test_coordinator_uses_one_owner_and_never_retries_unknown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    import astrolabe.evaluators.af3_lifecycle as lifecycle
    import astrolabe.interface as interface
    import astrolabe.protein_beam as beam

    config = tmp_path / "config.yaml"
    output = tmp_path / "output"
    service_root = tmp_path / "service"
    config.write_text(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "out_dir": str(output),
                    "seed": 7,
                    "protein_beam": {
                        "workers": 2,
                        "in_flight": 1,
                        "max_evaluations": 2,
                        "af3_call_budget": {"ledger": str(output / "calls.json")},
                    },
                }
            )
        ),
        encoding="utf-8",
    )
    owners = []
    calls = []

    class Owner:
        def __init__(self, cfg, destination, environment, *, settings=None):
            owners.append((cfg, destination, environment, settings))

        def __enter__(self):
            return "/harness/owner.sock"

        def __exit__(self, *args):
            return False

        def abort(self):
            calls.append(("abort",))

    def run_beam(cfg, destination, socket_path, **kwargs):
        calls.append((cfg, destination, socket_path, kwargs))
        return {"status": "complete"}

    monkeypatch.setattr(lifecycle, "OwnedAF3", Owner)
    monkeypatch.setattr(lifecycle, "recover_owned_services", lambda output: None)
    monkeypatch.setattr(interface, "cpu_coordinator_environment", lambda: nullcontext({"CUDA_VISIBLE_DEVICES": "0"}))
    monkeypatch.setattr(beam, "run_beam", run_beam)
    arguments = type(
        "Arguments",
        (),
        {
            "config": config,
            "output": output,
            "service_root": service_root,
            "resume": True,
            "source_fingerprint": harness._source_fingerprint(harness._source_manifest()),
        },
    )()
    assert harness._coordinator_main(arguments) == 0
    assert len(owners) == 1
    assert owners[0][1] == service_root
    transport = owners[0][3]
    assert type(transport).__name__ == "AF3ServiceSettings"
    assert transport.startup_timeout_s == 600.0
    assert transport.shutdown_timeout_s == 120.0
    assert transport.terminate_timeout_s == 10.0
    assert transport.max_pending == 4
    assert calls[0][3]["resume"] is True
    assert calls[0][3]["retry_interrupted"] is False
    assert callable(calls[0][3]["on_abort"])


def test_sitecustomize_publishes_durable_receipt_then_stops_only_its_group(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    instrumentation = tmp_path / "instrumentation"
    instrumentation.mkdir()
    (instrumentation / "sitecustomize.py").write_text(
        harness._instrumentation_source(), encoding="utf-8"
    )
    root_result = tmp_path / "run" / "candidates" / "000000" / "result.json"
    root_result.parent.mkdir(parents=True)
    receipt = tmp_path / "cut.receipt.json"
    token = "test-owned-token"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(instrumentation)
    environment.update(
        ASTROLABE_ACCEPTANCE_ROOT_RESULT=str(root_result),
        ASTROLABE_ACCEPTANCE_CUT_RECEIPT=str(receipt),
        ASTROLABE_ACCEPTANCE_CUT_TOKEN=token,
    )
    worker_code = (
        "import json, os\n"
        f"target={str(root_result)!r}\n"
        "temporary=target+'.tmp'\n"
        "open(temporary,'w').write(json.dumps({'id':0,'status':'done'}))\n"
        "os.replace(temporary,target)\n"
    )
    code = (
        "import subprocess, sys\n"
        f"worker={worker_code!r}\n"
        "raise SystemExit(subprocess.call([sys.executable, '-c', worker]))\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        env=environment,
        start_new_session=True,
    )
    try:
        identity = harness._process_identity(process.pid)
        assert identity is not None
        deadline = time.monotonic() + 10
        while not receipt.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert receipt.is_file()
        cut = json.loads(receipt.read_text())
        assert harness._validate_cut_receipt(
            cut,
            coordinator_pid=process.pid,
            root_result=root_result,
            token=token,
        )["id"] == 0
        assert harness._terminate_owned_group(process.pid, identity["start_ticks"])
        assert process.wait(timeout=10) in {-signal.SIGTERM, 128 + signal.SIGTERM}
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def test_cut_receipt_rejects_foreign_process_and_changed_root(tmp_path: Path) -> None:
    harness = _load_harness()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        root = tmp_path / "result.json"
        root.write_text(json.dumps({"id": 0, "status": "done"}), encoding="utf-8")
        identity = harness._process_identity(process.pid)
        assert identity is not None
        worker = dict(identity, boot_id=harness._boot_id())
        receipt = {
            "schema": "astrolabe.isolated-gpu-cut/v1",
            "token": "owned",
            "coordinator_pid": process.pid,
            "worker": worker,
            "root_result": str(root),
            "root_sha256": harness._sha256(root),
        }
        assert harness._validate_cut_receipt(
            receipt,
            coordinator_pid=process.pid,
            root_result=root,
            token="owned",
        )["id"] == 0
        with pytest.raises(harness.AcceptanceError, match="another harness"):
            harness._validate_cut_receipt(
                receipt,
                coordinator_pid=process.pid,
                root_result=root,
                token="foreign",
            )
        root.write_text(json.dumps({"id": 0, "status": "changed"}), encoding="utf-8")
        with pytest.raises(harness.AcceptanceError, match="changed"):
            harness._validate_cut_receipt(
                receipt,
                coordinator_pid=process.pid,
                root_result=root,
                token="owned",
            )
    finally:
        process.kill()
        process.wait()


def test_owned_group_requires_exact_identity_before_termination() -> None:
    harness = _load_harness()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        identity = harness._process_identity(process.pid)
        assert identity is not None
        assert not harness._terminate_owned_group(process.pid, "not-the-start")
        assert process.poll() is None
        assert harness._terminate_owned_group(process.pid, identity["start_ticks"])
        assert process.wait(timeout=10) in {-signal.SIGTERM, 128 + signal.SIGTERM}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_service_sibling_identity_is_verified_before_group_cleanup(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    service_root = tmp_path / "runs" / "recovery-service"
    lifecycle = service_root / "services" / "owned" / "lifecycle.json"
    process = subprocess.Popen(
        [sys.executable, "-c", "import signal; signal.pause()"],
        start_new_session=True,
    )
    try:
        identity = harness._process_identity(process.pid)
        assert identity is not None
        lifecycle.parent.mkdir(parents=True)
        lifecycle.write_text(
            json.dumps(
                {
                    "status": "ready",
                    "pid": process.pid,
                    "pid_start_ticks": identity["start_ticks"],
                    "boot_id": harness._boot_id(),
                }
            ),
            encoding="utf-8",
        )
        evidence = harness._service_evidence(service_root)
        assert evidence[0]["pid"] == process.pid
        assert evidence[0]["process_group_members"]
        harness._terminate_owned_services(service_root)
        harness._wait_owned_services_ended(service_root, timeout_s=10)
        assert process.wait(timeout=10) in {
            -signal.SIGTERM,
            128 + signal.SIGTERM,
        }
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("failure", ["wait", "validation"])
def test_interrupted_failure_always_cleans_verified_owned_group(
    tmp_path: Path,
    monkeypatch,
    failure: str,
) -> None:
    harness = _load_harness()
    out_root = tmp_path / "acceptance"
    recovery_output = out_root / "runs" / "recovery"
    pid_file = tmp_path / "owned-pids.txt"
    prepared = harness.PreparedAcceptance(
        out_root=out_root,
        case_config=tmp_path / "case.yaml",
        source_fingerprint="source",
        baseline_config=tmp_path / "baseline.yaml",
        recovery_config=tmp_path / "recovery.yaml",
        baseline_output=out_root / "runs" / "baseline",
        recovery_output=recovery_output,
        instrumentation=out_root / "instrumentation",
        baseline_service_root=out_root / "runs" / "baseline-service",
        recovery_service_root=out_root / "runs" / "recovery-service",
    )
    coordinator_code = (
        "import os, subprocess, sys, time\n"
        "child=subprocess.Popen([sys.executable,'-c','import signal; signal.pause()'])\n"
        f"open({str(pid_file)!r},'w').write(f'{{os.getpid()}} {{child.pid}}')\n"
        "child.wait()\n"
    )
    monkeypatch.setattr(
        harness,
        "_coordinator_command",
        lambda **kwargs: [sys.executable, "-c", coordinator_code],
    )
    def fail_after_start(*args, **kwargs):
        deadline = time.monotonic() + 5
        while not pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.02)
        if failure == "wait":
            raise harness.AcceptanceError("injected receipt wait failure")
        return {}

    monkeypatch.setattr(harness, "_wait_for_cut", fail_after_start)
    with pytest.raises(harness.AcceptanceError):
        harness._run_interrupted(prepared, timeout_s=5)
    supervisor_pid, child_pid = map(int, pid_file.read_text().split())
    deadline = time.monotonic() + 5
    while (
        any(harness._process_identity(pid) for pid in (supervisor_pid, child_pid))
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    assert harness._process_identity(supervisor_pid) is None
    assert harness._process_identity(child_pid) is None


def test_exact_recovery_comparison_keeps_sequence_float_metrics_and_coordinates(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    baseline_output, baseline_config = _write_native_run(tmp_path / "baseline")
    recovery_output, recovery_config = _write_native_run(tmp_path / "recovery")
    baseline = harness._native_snapshot(baseline_output, baseline_config)
    recovery = harness._native_snapshot(recovery_output, recovery_config)
    comparison = harness.compare_runs(
        baseline,
        recovery,
        baseline_config=baseline_config,
        recovery_config=recovery_config,
    )
    assert comparison["status"] == "PASS", comparison
    assert comparison["native_call_count"] == 2
    assert comparison["exact_metrics"] is True
    assert comparison["exact_coordinates"] is True
    assert comparison["exact_raw_score_arrays"] is True
    assert comparison["exact_inference_events"] is True
    assert recovery["inference_events"]["counts"] == {
        "request_received": 2,
        "gpu_start": 2,
        "gpu_end": 2,
        "gpu_ok": 2,
    }
    assert all(
        candidate["score_arrays"]["array_count"] >= 3
        for candidate in recovery["candidates"]
    )

    changed_output, changed_config = _write_native_run(
        tmp_path / "changed",
        x=1.2500001,
        score=0.1250001,
        score_array_value=0.7500001,
    )
    changed = harness._native_snapshot(changed_output, changed_config)
    failed = harness.compare_runs(
        baseline,
        changed,
        baseline_config=baseline_config,
        recovery_config=changed_config,
    )
    assert failed["status"] == "ISSUES"
    assert "raw score snapshot arrays differ" in failed["issues"]

    coordinate_output, coordinate_config = _write_native_run(
        tmp_path / "coordinate-changed",
        x=1.2500001,
        score=0.125,
        score_array_value=0.75,
    )
    coordinate_changed = harness._native_snapshot(
        coordinate_output, coordinate_config
    )
    coordinate_failed = harness.compare_runs(
        baseline,
        coordinate_changed,
        baseline_config=baseline_config,
        recovery_config=coordinate_config,
    )
    assert "candidate coordinates differ" in coordinate_failed["issues"]


def test_inference_evidence_is_read_from_service_sibling(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    output, config = _write_native_run(tmp_path / "run")
    service_root = output.parent / "run-service"
    service_root.mkdir()
    (output / "services").rename(service_root / "services")
    snapshot = harness._native_snapshot(output, config, service_root)
    assert snapshot["native_call_count"] == 2
    assert snapshot["inference_events"]["counts"] == {
        "request_received": 2,
        "gpu_start": 2,
        "gpu_end": 2,
        "gpu_ok": 2,
    }


def test_service_event_response_gap_fails_even_when_candidate_records_match(
    tmp_path: Path,
) -> None:
    harness = _load_harness()
    baseline_output, baseline_config = _write_native_run(tmp_path / "baseline")
    changed_output, changed_config = _write_native_run(tmp_path / "event-gap")
    owner_log = changed_output / "services" / "owned" / "owner.jsonl"
    events = [
        json.loads(line)
        for line in owner_log.read_text(encoding="utf-8").splitlines()
    ]
    events = [
        event
        for event in events
        if not (event["event"] == "gpu_end" and event["request_id"] == 2)
    ]
    owner_log.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    baseline = harness._native_snapshot(baseline_output, baseline_config)
    changed = harness._native_snapshot(changed_output, changed_config)
    comparison = harness.compare_runs(
        baseline,
        changed,
        baseline_config=baseline_config,
        recovery_config=changed_config,
    )
    assert changed["record_fresh_call_count"] == 2
    assert changed["native_call_count"] == 2
    assert "event requests lack one AF3 response completion each" in comparison["issues"]


def test_partial_config_or_source_change_is_detectable() -> None:
    harness = _load_harness()
    first = harness._source_fingerprint({"x": "1"})
    second = harness._source_fingerprint({"x": "2"})
    assert first != second
