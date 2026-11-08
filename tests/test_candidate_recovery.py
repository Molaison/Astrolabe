from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import pytest

import astrolabe.experimental.candidate_child as candidate_child
import astrolabe.experimental.candidate_batch_runner as runner
from astrolabe.design.collection_receipts import DurableCollectionTrial
from astrolabe.design.collection_search import CollectionAction
from astrolabe.design.runtime import ContractConflict, OutcomeUnknown
from astrolabe.design.runtime import Runtime
from astrolabe.design.types import key_for
from astrolabe.model_execution import capture_random_state


EXPECTED_JOBS = ("cand_0001", "cand_0002")


def _base_config(tmp_path: Path, *, seed: int = 16) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(
        "\n".join(
            [
                "task_name: base",
                f"out_dir: {tmp_path / 'out'}",
                f"seed: {seed}",
                "steps: 5",
                "seqs:",
                "  A: ACDEFGHIK",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _patch_identity(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_manifest=(),
    input_assets=None,
) -> None:
    assets = input_assets or {
        "schema": "astrolabe.test-input-assets/v1",
        "files": [],
    }
    monkeypatch.setattr(
        runner,
        "_source_manifest",
        lambda: tuple(source_manifest),
    )
    monkeypatch.setattr(runner, "_asset_manifest", lambda _cfg: dict(assets))
    monkeypatch.setattr(
        runner,
        "_scientific_config",
        lambda cfg: {
            "seed": int(cfg.seed),
            "steps": int(cfg.steps),
            "seqs": dict(cfg.get("seqs") or {}),
        },
    )


def _runner_script(tmp_path: Path, *, sleep_s: float = 0.02) -> tuple[Path, Path]:
    calls = tmp_path / "runner_calls.log"
    path = tmp_path / "fake_runner.py"
    path.write_text(
        f"""#!{sys.executable}
from pathlib import Path
import argparse
import os
import time
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("--standalone")
parser.add_argument("--no-chdir", action="store_true")
args = parser.parse_args()
cfg = OmegaConf.load(args.standalone)
descriptor = os.open({str(calls)!r}, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.write(descriptor, (str(cfg.task_name) + "\\n").encode("utf-8"))
os.close(descriptor)
time.sleep({sleep_s!r})
task_root = Path(cfg.out_dir) / str(cfg.task_name)
task_root.mkdir(parents=True, exist_ok=True)
score = 0.9 if str(cfg.task_name).endswith("cand_0001") else 0.5
(task_root / "scores.csv").write_text(
    "step,weighted_score,accepted,pA_sequence\\n"
    f"0,{{score}},true,ACDEFGHIK\\n",
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path, calls


def _config(
    tmp_path: Path,
    base: Path,
    fake_runner: Path,
    *,
    resume: bool = False,
) -> runner.ProcessCandidateRunnerConfig:
    return runner.ProcessCandidateRunnerConfig(
        config_yaml=base,
        out_root=tmp_path / "out",
        runner=fake_runner,
        num_candidates=2,
        min_completed=2,
        max_wait_s=5.0,
        max_in_flight=2,
        decision_rounds=1,
        candidate_steps=2,
        candidate_timeout_s=5.0,
        resume=resume,
    )


def _call_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def test_committed_round_resume_reuses_results_order_and_rng(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_config(tmp_path)
    fake_runner, calls = _runner_script(tmp_path)
    _patch_identity(monkeypatch)
    random.seed(41)
    np.random.seed(43)

    first = runner.run_process_candidate_batch(_config(tmp_path, base, fake_runner))
    rng_after_first = capture_random_state()
    assert first["selected_candidate_ids"] == ["cand_0001"]
    assert _call_count(calls) == 2

    random.seed(999)
    np.random.seed(999)
    resumed = runner.run_process_candidate_batch(
        _config(tmp_path, base, fake_runner, resume=True)
    )

    assert resumed["collections"] == first["collections"]
    assert resumed["selected_candidate_ids"] == ["cand_0001"]
    assert _call_count(calls) == 2
    assert key_for(capture_random_state()) == key_for(rng_after_first)


def test_committed_collection_reopens_without_reselecting_or_rerunning_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_config(tmp_path)
    fake_runner, calls = _runner_script(tmp_path)
    _patch_identity(monkeypatch)
    config = _config(tmp_path, base, fake_runner)
    worker = runner.ProcessCandidateWorker(config, runner._load_base_cfg(base))
    pool_config = runner.CandidatePoolConfig(
        num_candidates=2,
        min_completed=2,
        max_wait_s=5.0,
        max_in_flight=2,
    )
    submit = CollectionAction(
        kind="submit",
        generation=1,
        jobs=tuple(
            runner.CandidateTask(
                candidate_id=job_id,
                generation=1,
                payload={"sequences": {}},
            )
            for job_id in EXPECTED_JOBS
        ),
    )
    collect = CollectionAction(kind="collect", generation=1)
    runtime_root = tmp_path / "collection-runtime"

    with Runtime(root=runtime_root) as runtime:
        receipts = runner.ProcessCandidateReceipts(
            runtime,
            worker,
            resumed=False,
        )
        trial = DurableCollectionTrial(
            runtime=runtime,
            config=pool_config,
            worker=worker,
            selector=runner.select_highest_score_candidate,
            receipts=receipts,
            action_codec=runner._ACTION_CODEC,
            result_codec=runner._EVENT_CODEC,
        )
        trial.perform(submit)
        first = trial.perform(collect)
        expected_collection = first.collections[0]
        expected_rng = key_for(capture_random_state())
        trial.close()

    assert _call_count(calls) == 2
    random.seed(1001)
    np.random.seed(1002)
    with Runtime(root=runtime_root) as runtime:
        receipts = runner.ProcessCandidateReceipts(
            runtime,
            worker,
            resumed=True,
        )
        resumed = DurableCollectionTrial(
            runtime=runtime,
            config=pool_config,
            worker=worker,
            selector=runner.select_highest_score_candidate,
            receipts=receipts,
            action_codec=runner._ACTION_CODEC,
            result_codec=runner._EVENT_CODEC,
            resumed=True,
        )
        recovered = resumed.perform(collect)
        assert key_for(capture_random_state()) == expected_rng
        next_submit = CollectionAction(
            kind="submit",
            generation=2,
            jobs=tuple(
                runner.CandidateTask(
                    candidate_id=job_id,
                    generation=2,
                    payload={"sequences": {}},
                )
                for job_id in ("cand_0003", "cand_0004")
            ),
        )
        resumed.perform(next_submit)
        continued = resumed.perform(
            CollectionAction(kind="collect", generation=2)
        )
        resumed.close()

    assert recovered.collections == (expected_collection,)
    assert continued.collections[0].collection_index == 2
    assert _call_count(calls) == 4


def test_completed_child_without_collection_commit_stops_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_config(tmp_path)
    fake_runner, calls = _runner_script(tmp_path)
    _patch_identity(monkeypatch)
    config = _config(tmp_path, base, fake_runner)
    worker = runner.ProcessCandidateWorker(config, runner._load_base_cfg(base))
    task = runner.CandidateTask(
        candidate_id="cand_0001",
        generation=1,
        payload={"sequences": {}},
    )
    runtime_root = tmp_path / "uncommitted-runtime"
    with Runtime(root=runtime_root) as runtime:
        receipts = runner.ProcessCandidateReceipts(
            runtime,
            worker,
            resumed=False,
        )
        assert receipts.admission(task) is None
        result = worker(task)
        receipts.terminal_observed(task, result)

    assert _call_count(calls) == 1
    with Runtime(root=runtime_root) as runtime:
        receipts = runner.ProcessCandidateReceipts(
            runtime,
            worker,
            resumed=True,
        )
        trial = DurableCollectionTrial(
            runtime=runtime,
            config=runner.CandidatePoolConfig(
                num_candidates=2,
                min_completed=2,
                max_wait_s=5.0,
                max_in_flight=2,
            ),
            worker=worker,
            selector=runner.select_highest_score_candidate,
            receipts=receipts,
            action_codec=runner._ACTION_CODEC,
            result_codec=runner._EVENT_CODEC,
            resumed=True,
        )
        with pytest.raises(OutcomeUnknown, match="selection boundary"):
            trial.perform(CollectionAction(kind="collect", generation=1))
        trial.close()
    assert _call_count(calls) == 1


def test_parent_kill_preserves_independent_child_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_config(tmp_path)
    fake_runner, calls = _runner_script(tmp_path, sleep_s=0.7)
    _patch_identity(monkeypatch)
    driver = tmp_path / "parent_driver.py"
    driver.write_text(
        f"""import os
from pathlib import Path
from omegaconf import OmegaConf
import astrolabe.experimental.candidate_batch_runner as runner
runner._source_manifest = lambda: ()
runner._asset_manifest = lambda _cfg: {{
    "schema": "astrolabe.test-input-assets/v1",
    "files": [],
}}
runner._scientific_config = lambda cfg: {{
    "seed": int(cfg.seed),
    "steps": int(cfg.steps),
    "seqs": dict(cfg.get("seqs") or {{}}),
}}
cfg = runner.ProcessCandidateRunnerConfig(
    config_yaml=Path({str(base)!r}),
    out_root=Path({str(tmp_path / "out")!r}),
    runner=Path({str(fake_runner)!r}),
    num_candidates=1,
    min_completed=1,
    max_wait_s=5.0,
    decision_rounds=1,
    candidate_steps=2,
    candidate_timeout_s=5.0,
)
worker = runner.ProcessCandidateWorker(cfg, OmegaConf.load(cfg.config_yaml))
worker(runner.CandidateTask(candidate_id="cand_0001", generation=1))
""",
        encoding="utf-8",
    )
    parent = subprocess.Popen(
        [sys.executable, str(driver)],
        cwd=runner._repo_root(),
        start_new_session=True,
    )
    receipt_path = tmp_path / "out" / "candidate_pool" / "cand_0001" / (
        "candidate_receipt.json"
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if receipt_path.is_file():
            receipt = runner._read_candidate_receipt(receipt_path)
            if receipt.status == "issued":
                break
        time.sleep(0.02)
    else:
        parent.kill()
        parent.wait()
        pytest.fail("candidate child never reached issued")

    os.kill(parent.pid, signal.SIGKILL)
    parent.wait(timeout=5.0)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        receipt = runner._read_candidate_receipt(receipt_path)
        if receipt.status in {"completed", "failed", "unknown"}:
            break
        time.sleep(0.02)
    assert receipt.status == "completed"
    assert _call_count(calls) == 1

    worker = runner.ProcessCandidateWorker(
        runner.ProcessCandidateRunnerConfig(
            config_yaml=base,
            out_root=tmp_path / "out",
            runner=fake_runner,
            candidate_steps=2,
        ),
        runner._load_base_cfg(base),
    )
    result = worker(runner.CandidateTask(candidate_id="cand_0001", generation=1))
    assert result.status == "completed"
    assert _call_count(calls) == 1


def test_issued_without_terminal_proof_is_not_reissued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_config(tmp_path)
    fake_runner, calls = _runner_script(tmp_path)
    _patch_identity(monkeypatch)
    worker = runner.ProcessCandidateWorker(
        _config(tmp_path, base, fake_runner),
        runner._load_base_cfg(base),
    )
    task = runner.CandidateTask(candidate_id="cand_0001", generation=1)
    receipt = worker.prepare(task)
    runner._write_candidate_receipt(
        Path(receipt.config_yaml).parent / "candidate_receipt.json",
        replace(receipt, status="issued"),
    )

    with pytest.raises(OutcomeUnknown, match="no terminal owner proof"):
        worker(task)

    assert _call_count(calls) == 0
    with Runtime(root=tmp_path / "runtime") as runtime:
        receipts = runner.ProcessCandidateReceipts(
            runtime,
            worker,
            resumed=True,
        )
        with pytest.raises(OutcomeUnknown, match="without terminal proof"):
            receipts.admission(task)
        child_task = runner.CandidateTask(
            candidate_id=receipt.job_id,
            generation=receipt.generation,
            payload={"sequences": receipt.sequences},
        )
        completed_result = runner.CandidateResult(
            task=child_task,
            status="completed",
            score=0.7,
            payload=runner.ProcessCandidatePayload(
                config_yaml=receipt.config_yaml,
                out_dir=receipt.out_dir,
                task_name=receipt.task_name,
                stdout_log=receipt.stdout_log,
                stderr_log=receipt.stderr_log,
                sequences={"A": "AAAA"},
            ),
        )
        runner._write_candidate_receipt(
            Path(receipt.config_yaml).parent / "candidate_receipt.json",
            replace(
                receipt,
                status="completed",
                result=completed_result,
                cli_provenance={"returncode": 0, "timed_out": False},
                output_manifest=(
                    {"path": "scores.csv", "sha256": "0" * 64, "size_bytes": 1},
                ),
            ),
        )
        assert receipts.admission(child_task) == completed_result
    assert _call_count(calls) == 0


def test_candidate_child_uses_job_scoped_capture_and_claim_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_config(tmp_path)
    fake_runner, _calls = _runner_script(tmp_path)
    _patch_identity(monkeypatch)
    config = _config(tmp_path, base, fake_runner)
    worker = runner.ProcessCandidateWorker(config, runner._load_base_cfg(base))
    task = runner.CandidateTask(
        candidate_id="cand_0001",
        generation=1,
        payload={"sequences": {}},
    )
    receipt = worker.prepare(task)
    task_root = tmp_path / "out" / receipt.task_name
    task_root.mkdir(parents=True)
    (task_root / "scores.csv").write_text(
        "step,weighted_score,accepted,pA_sequence\n0,0.7,true,AAAA\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    class _Process:
        pid = os.getpid()
        returncode = 0

        def wait(self, timeout=None):
            captured["wait_timeout"] = timeout
            return 0

    def _popen(_command, **kwargs):
        captured.update(kwargs)
        return _Process()

    monkeypatch.setattr(candidate_child.subprocess, "Popen", _popen)
    monkeypatch.setenv(
        "ASTROLABE_ACCEPTANCE_CAPTURE_ROOT",
        str(tmp_path / "captures"),
    )
    assert (
        candidate_child.run(
            receipt_path=Path(receipt.config_yaml).parent
            / "candidate_receipt.json",
            runner=fake_runner,
            timeout_s=None,
        )
        == 0
    )

    child_env = captured["env"]
    assert Path(child_env["ASTROLABE_ACCEPTANCE_CAPTURE"]).parent.name == (
        "cand_0001"
    )
    assert Path(child_env["ASTROLABE_ACCEPTANCE_CLAIMS"]).parent.name == (
        "cand_0001"
    )
    assert child_env["ASTROLABE_CANDIDATE_ID"] == "cand_0001"
    assert (
        runner._read_candidate_receipt(
            Path(receipt.config_yaml).parent / "candidate_receipt.json"
        ).status
        == "completed"
    )


def test_admitted_attempt_has_exclusive_outer_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_config(tmp_path)
    fake_runner, _calls = _runner_script(tmp_path)
    _patch_identity(monkeypatch)
    config = _config(tmp_path, base, fake_runner)
    worker = runner.ProcessCandidateWorker(config, runner._load_base_cfg(base))
    task = runner.CandidateTask(
        candidate_id="cand_0001",
        generation=1,
        payload={"sequences": {}},
    )
    worker.prepare(task)
    started = threading.Event()
    both_claims_started = threading.Event()
    release = threading.Event()
    invocation_lock = threading.Lock()
    invocations = 0
    claim_invocations = 0

    def _claim(receipt):
        nonlocal claim_invocations
        with invocation_lock:
            claim_invocations += 1
            if claim_invocations == 2:
                both_claims_started.set()
        return runner.ProcessCandidateWorker._claim_attempt(receipt)

    def _fake_run(command, **_kwargs):
        nonlocal invocations
        with invocation_lock:
            invocations += 1
        started.set()
        release.wait(timeout=5.0)
        receipt_path = Path(command[command.index("--receipt") + 1])
        receipt = runner._read_candidate_receipt(receipt_path)
        result = runner.CandidateResult(
            task=task,
            status="completed",
            score=0.7,
            payload=runner.ProcessCandidatePayload(
                config_yaml=receipt.config_yaml,
                out_dir=receipt.out_dir,
                task_name=receipt.task_name,
                stdout_log=receipt.stdout_log,
                stderr_log=receipt.stderr_log,
                sequences={"A": "AAAA"},
            ),
        )
        runner._write_candidate_receipt(
            receipt_path,
            replace(
                receipt,
                status="completed",
                result=result,
                cli_provenance={"returncode": 0, "timed_out": False},
                output_manifest=(
                    {"path": "scores.csv", "sha256": "0" * 64, "size_bytes": 1},
                ),
            ),
        )
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)
    monkeypatch.setattr(worker, "_claim_attempt", _claim)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(worker, task)
        assert started.wait(timeout=5.0)
        second = executor.submit(worker, task)
        assert both_claims_started.wait(timeout=5.0)
        release.set()
        outcomes = []
        for future in (first, second):
            try:
                outcomes.append(future.result(timeout=5.0))
            except OutcomeUnknown as error:
                outcomes.append(error)

    assert invocations == 1
    assert sum(
        isinstance(item, runner.CandidateResult)
        and item.status == "completed"
        for item in outcomes
    ) == 1
    assert sum(isinstance(item, OutcomeUnknown) for item in outcomes) == 1


def test_timeout_is_durable_unknown_and_is_not_reissued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_config(tmp_path)
    fake_runner, calls = _runner_script(tmp_path, sleep_s=2.0)
    _patch_identity(monkeypatch)
    config = replace(
        _config(tmp_path, base, fake_runner),
        candidate_timeout_s=0.7,
    )
    worker = runner.ProcessCandidateWorker(
        config,
        runner._load_base_cfg(base),
    )
    task = runner.CandidateTask(candidate_id="cand_0001", generation=1)

    with pytest.raises(OutcomeUnknown):
        worker(task)
    receipt = worker.receipt(task)
    assert receipt is not None
    assert receipt.status == "unknown"
    assert receipt.result is None
    assert _call_count(calls) == 1

    with pytest.raises(OutcomeUnknown, match="no terminal owner proof"):
        worker(task)
    assert _call_count(calls) == 1


@pytest.mark.parametrize("changed", ["config", "source", "weight"])
def test_same_path_content_change_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    base = _base_config(tmp_path)
    fake_runner, _calls = _runner_script(tmp_path)
    source_file = tmp_path / "source.py"
    weight_file = tmp_path / "weight.bin"
    source_file.write_text("source-v1\n", encoding="utf-8")
    weight_file.write_bytes(b"weight-v1")

    def _source():
        return (
            {
                "path": source_file.name,
                "sha256": runner._sha256_file(source_file),
                "size_bytes": source_file.stat().st_size,
            },
        )

    def _assets(_cfg):
        return {
            "schema": "astrolabe.test-input-assets/v1",
            "files": [
                {
                    "path": str(weight_file),
                    "sha256": runner._sha256_file(weight_file),
                    "size_bytes": weight_file.stat().st_size,
                }
            ],
        }

    monkeypatch.setattr(runner, "_source_manifest", _source)
    monkeypatch.setattr(runner, "_asset_manifest", _assets)
    monkeypatch.setattr(
        runner,
        "_scientific_config",
        lambda cfg: {"seed": int(cfg.seed), "steps": int(cfg.steps)},
    )
    config = _config(tmp_path, base, fake_runner)
    worker = runner.ProcessCandidateWorker(config, runner._load_base_cfg(base))
    task = runner.CandidateTask(candidate_id="cand_0001", generation=1)
    worker.prepare(task)

    if changed == "config":
        base.write_text(base.read_text(encoding="utf-8") + "extra: 1\n", encoding="utf-8")
    elif changed == "source":
        source_file.write_text("source-v2\n", encoding="utf-8")
    else:
        weight_file.write_bytes(b"weight-v2")

    changed_worker = runner.ProcessCandidateWorker(
        config,
        runner._load_base_cfg(base),
    )
    with pytest.raises(ContractConflict, match="changed at"):
        changed_worker.prepare(task)


def test_completion_pool_keeps_normal_pool_concurrency() -> None:
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def _worker(task: runner.CandidateTask) -> runner.CandidateResult:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.12)
        with lock:
            active -= 1
        return runner.CandidateResult(task=task, status="completed", score=0.5)

    started = time.monotonic()
    with runner.CandidatePoolScheduler(
        runner.CandidatePoolConfig(
            num_candidates=2,
            min_completed=2,
            max_wait_s=2.0,
            max_in_flight=2,
        ),
        _worker,
    ) as pool:
        pool.submit_generation(
            1,
            [
                runner.CandidateTask(candidate_id="cand_0001", generation=1),
                runner.CandidateTask(candidate_id="cand_0002", generation=1),
            ],
        )
        collection = pool.collect_until_ready(target_generation=1)
    elapsed = time.monotonic() - started

    assert collection is not None
    assert maximum_active == 2
    assert elapsed < 0.22


def test_rehydrated_ready_result_is_idempotent_during_submit_recovery() -> None:
    job = runner.CandidateTask(candidate_id="cand_0001", generation=1)
    saved = runner.CandidateResult(task=job, status="completed", score=0.7)

    with runner.CandidatePoolScheduler(
        runner.CandidatePoolConfig(
            num_candidates=1,
            min_completed=1,
            max_wait_s=1.0,
        ),
        lambda _job: pytest.fail("saved candidate must not run"),
        admission=lambda admitted: saved if admitted == job else None,
    ) as pool:
        pool.rehydrate(ready_results=(saved,), collection_index=3)
        pool.submit_generation(1, [job])
        collection = pool.collect_until_ready(target_generation=1)

    assert collection is not None
    assert collection.collection_index == 4
    assert collection.results == (saved,)


def test_drain_with_ready_and_issued_jobs_stops_before_selection() -> None:
    ready_job = runner.CandidateTask(candidate_id="ready", generation=1)
    issued_job = runner.CandidateTask(candidate_id="issued", generation=1)
    ready = runner.CandidateResult(task=ready_job, status="completed", score=0.7)
    commits = []

    with runner.CandidatePoolScheduler(
        runner.CandidatePoolConfig(
            num_candidates=2,
            min_completed=1,
            max_wait_s=1.0,
        ),
        lambda _job: pytest.fail("restored jobs must not run"),
        collection_commit=commits.append,
    ) as pool:
        pool.rehydrate(
            ready_results=(ready,),
            issued_jobs=(issued_job,),
            collection_index=2,
        )
        with pytest.raises(OutcomeUnknown, match="no owner terminal proof"):
            pool.drain()

    assert commits == []
    assert pool._collection_index == 2


def test_submit_recovery_enables_collect_for_the_same_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_config(tmp_path)
    fake_runner, calls = _runner_script(tmp_path)
    _patch_identity(monkeypatch)
    config = _config(tmp_path, base, fake_runner)
    worker = runner.ProcessCandidateWorker(config, runner._load_base_cfg(base))
    pool_config = runner.CandidatePoolConfig(
        num_candidates=2,
        min_completed=2,
        max_wait_s=5.0,
        max_in_flight=2,
    )
    submit = CollectionAction(
        kind="submit",
        generation=1,
        jobs=tuple(
            runner.CandidateTask(
                candidate_id=job_id,
                generation=1,
                payload={"sequences": {}},
            )
            for job_id in EXPECTED_JOBS
        ),
    )
    with Runtime(root=tmp_path / "submit-recovery-runtime") as runtime:
        runtime.save("control/contract", key_for(("test-control/v1",)))
        receipts = runner.ProcessCandidateReceipts(
            runtime,
            worker,
            resumed=True,
        )
        trial = DurableCollectionTrial(
            runtime=runtime,
            config=pool_config,
            worker=worker,
            selector=runner.select_highest_score_candidate,
            receipts=receipts,
            action_codec=runner._ACTION_CODEC,
            result_codec=runner._EVENT_CODEC,
            resumed=True,
        )
        trial.recover(submit)
        collection = trial.perform(
            CollectionAction(kind="collect", generation=1)
        )
        trial.close()

    assert collection.collections[0].collection_index == 1
    assert _call_count(calls) == 2


@pytest.mark.parametrize("trigger", ["min_completed", "max_wait_s"])
def test_collection_commit_records_consumed_and_carryforward_without_reselection(
    trigger: str,
) -> None:
    commits = []

    def _worker(task: runner.CandidateTask) -> runner.CandidateResult:
        delay = 0.01 if task.candidate_id == "fast" else 0.2
        time.sleep(delay)
        return runner.CandidateResult(
            task=task,
            status="completed",
            score=0.7 if task.candidate_id == "fast" else 0.9,
        )

    max_wait_s = 0.05 if trigger == "max_wait_s" else 1.0
    with runner.CandidatePoolScheduler(
        runner.CandidatePoolConfig(
            num_candidates=2,
            min_completed=2,
            max_wait_s=max_wait_s,
            max_in_flight=2,
        ),
        _worker,
        collection_commit=commits.append,
    ) as pool:
        pool.submit_generation(
            1,
            [
                runner.CandidateTask(candidate_id="fast", generation=1),
                runner.CandidateTask(candidate_id="slow", generation=1),
            ],
        )
        first = pool.collect_until_ready(target_generation=1)
        drained = pool.drain()

    assert first is not None
    assert commits[0].trigger == trigger
    if trigger == "min_completed":
        assert commits[0].consumed_job_ids == ("fast", "slow")
        assert commits[0].carryforward_job_ids == ()
        assert drained == []
    else:
        assert commits[0].consumed_job_ids == ("fast",)
        assert commits[0].carryforward_job_ids == ("slow",)
        assert len(drained) == 1
        assert commits[1].consumed_job_ids == ("slow",)
        assert commits[1].trigger == "drain"
