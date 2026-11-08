from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing
import os
import random
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from astrolabe.app import Astrolabe
from astrolabe.core import global_config
from astrolabe.design.branch_types import BranchAction, BranchEvent, BranchJob, BranchProposal, BranchTerminal, BranchingState, branch_codec
from astrolabe.design.branching import BranchingPolicy
from astrolabe.design.composition import ControlRun
from astrolabe.design.native_scores import snapshot_scores
from astrolabe.design.native_refinement import NativeMeasurement
from astrolabe.design.native_refinement_runtime import ScoreReceipt, _SCORE_RECEIPT_CODEC
from astrolabe.design.parallel_refinement import ParallelRefinementOwner, _PredictionPool, _ScoreHandoff
from astrolabe.design.runtime import ContractConflict
from astrolabe.design.store import DataStore, JSONCodec, design_codec
from astrolabe.design.types import History, Key, key_for
from astrolabe.evaluators.utils import StandardScores
from test_native_refinement_recovery import _case, _label, _seed, PROTOCOL


@pytest.mark.parametrize("returned_client", [True, False])
def test_shared_registry_prepares_only_explicit_lazy_client(tmp_path, monkeypatch, returned_client):
    from astrolabe.design.parallel_refinement import _shared_models
    from astrolabe.model_execution import NativeModelClient

    monkeypatch.delenv("ASTROLABE_RESIDENT_MODEL_REGISTRY", raising=False)
    client = NativeModelClient.__new__(NativeModelClient)
    client.owner = SimpleNamespace(socket_path=tmp_path / "native.sock", identity="native-model",
                                   spec=SimpleNamespace(backend="mol1"))
    client.metadata = {"runtime_lifecycle": {"weights_loaded": True}}
    calls = []

    class LazyRunner:
        _client = None

        def prepare_shared_model(self):
            calls.append("prepared")
            self._client = client
            return client if returned_client else None

    evaluator = SimpleNamespace(runner=LazyRunner(), ignored=SimpleNamespace(nested=client))
    owner = SimpleNamespace(models=SimpleNamespace(get=lambda: SimpleNamespace(evaluator=evaluator)),
                            worker_output=tmp_path / "workers", spec=SimpleNamespace(event=lambda *a, **kw: None))
    with _shared_models(owner, True):
        path = Path(os.environ["ASTROLABE_RESIDENT_MODEL_REGISTRY"])
        assert json.loads(path.read_text())["models"] == {
            "native-model": {"backend": "mol1", "socket": str(client.owner.socket_path)},
        }
    assert calls == ["prepared"]
    assert "ASTROLABE_RESIDENT_MODEL_REGISTRY" not in os.environ


@pytest.fixture(autouse=True)
def clean_globals():
    global_config.reset()
    yield
    global_config.reset()


def test_lazy_proposals_advance_on_parent_with_original_scope_and_wait_budget(tmp_path):
    data = DataStore(tmp_path / "data")
    proposal_codec, job_codec = branch_codec(BranchProposal), branch_codec(BranchJob)
    for codec in (proposal_codec, job_codec, JSONCodec()):
        data.register(codec)
    history = History(key_for("admission-history"), ())
    runtime = SimpleNamespace(_work="admission", history=history)
    calls = []
    runtime._capture_rng = lambda: data.put({"mutations": len(calls)}, JSONCodec())
    value = data.put({}, JSONCodec())
    remote_future = Future()
    caller = threading.get_ident()

    def terminal(job):
        calls.append((job.id, runtime._work, runtime.history, threading.get_ident()))
        assert not remote_future.done()
        return BranchTerminal(job, "completed", BranchProposal(
            job.id, 0, f"proposal-{job.id}", value,
        ))

    owner = SimpleNamespace(data=data, job_codec=job_codec, proposal_codec=proposal_codec,
                            runtime=runtime, _terminal=terminal)
    pool = _PredictionPool(owner, SimpleNamespace(submit=lambda *args: remote_future))
    futures = [pool.submit(None, {
        "kind": "propose", "job": asdict(data.put(BranchJob("propose", i, 1, i), job_codec)),
    }) for i in (1, 2, 3)]
    assert not any(future.done() for future in futures)
    assert pool.submit(None, {"kind": "evaluate"}) is remote_future
    pool.advance(BranchAction(BranchingState(), wait="none"))
    assert calls == []
    later_history = History(key_for("collection-history"), ())
    runtime._work, runtime.history = "collection", later_history
    pool.advance(BranchAction(BranchingState(), wait="any"))
    assert [future.done() for future in futures] == [True, False, False]
    pool.advance(BranchAction(BranchingState(), wait="all"))
    assert all(future.done() for future in futures)
    assert all("after_rng" in future.result() for future in futures)
    assert calls == [(i, "admission", history, caller) for i in (1, 2, 3)]
    assert (runtime._work, runtime.history) == ("collection", later_history)


def test_score_handoff_waits_for_score_and_forwards_failures():
    predictions, scores, submitted = [], [], []

    def predict(_worker, task):
        future = Future()
        predictions.append(future)
        return future

    def score(_worker, task):
        future = Future()
        scores.append(future)
        submitted.append(task)
        return future

    handoff = _ScoreHandoff(SimpleNamespace(submit=predict),
                            SimpleNamespace(submit=score), object())
    task = {"id": 1}
    final = handoff.submit(object(), task)
    assert final.running() and not final.cancel()
    assert not final.done() and not scores
    predictions[0].set_result({"raw": "durable"})
    assert submitted == [task] and not final.done()
    scores[0].set_result({"score": "complete"})
    handoff.drain()
    assert final.result() == {"score": "complete"}

    failed_prediction = handoff.submit(object(), task)
    predictions[1].set_exception(RuntimeError("prediction failed"))
    assert str(failed_prediction.exception()) == "prediction failed"
    assert len(scores) == 1

    failed_score = handoff.submit(object(), task)
    predictions[2].set_result({"raw": "durable"})
    scores[1].set_exception(RuntimeError("Score failed"))
    handoff.drain()
    assert str(failed_score.exception()) == "Score failed"


def _owner(case, pool, worker):
    case.spec = replace(case.spec, width=2, proposals_per_parent=2)
    global_config.init_const("raw_model_results_dir", case.root / "parent-raw")
    global_config.init_const("out_dir", case.root / "parent-output")
    return ParallelRefinementOwner(
        case.spec, case.runtime, protocol=PROTOCOL, pool=pool, worker=worker,
        worker_output=case.root / "workers",
    )


def _scored_result(case, task, raw):
    from astrolabe.design.native_scores import restore_scores
    from astrolabe.design.refinement_worker import _blob, _sidecars
    root = Path(raw["job_raw_dir"]).parent / "score"
    root.mkdir()
    (root / "derived.txt").write_text("measurement")
    scores = restore_scores(_blob(raw["raw_scores"]), case.data, root / "structures")
    measurement = NativeMeasurement(
        float(np.asarray(scores.plddt["A"][0]).mean()),
        {"active_sequence": task["sequence"]}, str(scores.complex_paths.get("A", "")),
    )
    measured = snapshot_scores(scores, case.data)
    receipt = case.data.put(ScoreReceipt(
        Key(task["_request_hash"]), _blob(raw["raw_scores"]), PROTOCOL,
        measured, measurement, _blob(raw["prediction_after_rng"]),
    ), _SCORE_RECEIPT_CODEC)
    case.data.persist((receipt,))
    return {**raw, "score_receipt": asdict(receipt), "score_worker_pid": raw["worker_pid"],
            "score_raw_dir": str(root), "score_sidecars": _sidecars(root)}


def _run(owner, *, pipeline=False):
    policy = BranchingPolicy(
        width=2, proposals_per_parent=2, max_evaluations=3, proposal_cap=8,
        seed=owner.spec.seed, evaluation_seed=owner.spec.seed,
        proposal_seed=_seed, proposal_seed_identity="test-proposal-seed",
        quota_mode="attempts", expansion="round_robin",
        proposal_batch_size=4, evaluation_width=2,
        pipeline=pipeline,
    )
    owner.bootstrap()
    initial = policy.start(owner.root_job(protocol=PROTOCOL))
    return Astrolabe(ControlRun(
        policy, owner, initial, owner.runtime, branch_codec(type(initial)),
        branch_codec(BranchAction), branch_codec(BranchEvent),
    )).run()


@pytest.mark.parametrize("slow_job", [1, 2])
@pytest.mark.parametrize("pipeline", [False, True])
def test_batch_predictions_overlap_but_candidates_and_measurements_are_isolated(tmp_path, slow_job, pipeline):
    case = _case(tmp_path, max_evaluations=3)
    def forbidden_parent_loss(*args):
        raise AssertionError("parent must not compute CPU loss")

    case.spec = replace(case.spec, measure=forbidden_parent_loss)
    after_rng = case.runtime._capture_rng()
    case.data.persist((after_rng,))
    rng = case.data.get(after_rng, design_codec)
    after_by_id = {
        identifier: case.data.put({**rng, "python": random.Random(identifier + 100).getstate()}, design_codec)
        for identifier in range(3)
    }
    case.data.persist(tuple(after_by_id.values()))
    barrier = threading.Barrier(2)
    first_prediction_started = threading.Event()
    fast_finished = threading.Event()
    calls = []
    lock = threading.Lock()
    first_prediction = []
    original_propose = case.spec.propose

    def propose(*args, **kwargs):
        if pipeline and kwargs["step"] == 2:
            assert first_prediction and not first_prediction[0].done()
            assert first_prediction_started.wait(timeout=10)
        return original_propose(*args, **kwargs)

    case.spec = replace(case.spec, propose=propose)

    def remote_prediction(task):
        # A remote DTO fixture, not a thread-safe legacy evaluator claim.
        if pipeline and task["id"] == 1:
            assert len([row for row in case.recorder.calls if row[0] == "propose"]) == 1
            first_prediction_started.set()
        if task["id"]:
            barrier.wait(timeout=10)
            if task["id"] == slow_job:
                assert fast_finished.wait(timeout=10)
        raw_dir = tmp_path / "workers" / str(task["id"]) / "raw"
        raw_dir.mkdir(parents=True)
        sidecar = raw_dir / "confidence.json"
        content = task["sequence"].encode()
        sidecar.write_bytes(content)
        scores = StandardScores(
            names=["A"], plddt={"A": [np.asarray([0.4, 0.8], dtype=np.float32)]},
            complex_paths={"A": sidecar},
        )
        reference = snapshot_scores(scores, case.data)
        case.data.persist((reference,))
        with lock:
            calls.append(task["id"])
        result = _scored_result(case, task, {
            **task, "raw_scores": asdict(reference), "worker_pid": task["id"] + 100,
            "constructor_count": 1, "prediction_wall_s": 0.0,
            "prediction_after_rng": asdict(after_by_id[task["id"]]),
            "job_raw_dir": str(raw_dir),
            "sidecars": [{
                "path": str(sidecar), "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }],
        })
        if task["id"] and task["id"] != slow_job:
            fast_finished.set()
        return result

    try:
        with ThreadPoolExecutor(2) as pool:
            class ObservedPool:
                def submit(self, worker, task):
                    if pipeline and task["id"] == 1:
                        assert len([row for row in case.recorder.calls if row[0] == "propose"]) == 1
                    future = pool.submit(worker, task)
                    if task["id"] == 1:
                        first_prediction.append(future)
                    return future

            owner = _owner(case, ObservedPool(), remote_prediction)
            final = _run(owner, pipeline=pipeline)
            assert sorted(calls) == [0, 1, 2]
            assert calls == [0, 3 - slow_job, slow_job]
            assert case.runtime._capture_rng() == after_by_id[2]
            assert len(final.records) == 3
            assert len(final.frontier) == 2
            assert final.proposals == 2
            assert [owner.metrics(owner.member(member))["active_sequence"]
                    for member in final.records] == ["AC", "GC", "GU"]
            assert not any(row[0] == "measure" for row in case.recorder.calls)
            assert global_config.protomers["A"].sequence == "AC"
            assert global_config.raw_model_results_dir == tmp_path / "parent-raw"
            assert global_config.out_dir == tmp_path / "parent-output"
            assert len(case.recorder.loads) == 2
            assert all(row[0] != "evaluate" for row in case.recorder.calls)
            root = owner.root_job(protocol=PROTOCOL)
            receipt = owner.trial._receipt(root)
            raw = owner.data.get(receipt.result, JSONCodec())
            assert not (Path(raw["job_raw_dir"]) / "derived.txt").exists()
            assert (Path(raw["score_raw_dir"]) / "derived.txt").is_file()
            before = len([row for row in case.recorder.calls if row[0] == "measure"])
            # Simulate delivery after measurement committed, before the outer
            # control receipt did. Recovery reuses the raw result and measure.
            with case.runtime.bind("test-redelivery", History(key_for(()), ())):
                owner._result(root, raw)
            assert len([row for row in case.recorder.calls if row[0] == "measure"]) == before
            assert sorted(calls) == [0, 1, 2]
            owner.shutdown()
    finally:
        case.close()


def test_sidecar_change_prevents_measurement_without_repeating_prediction(tmp_path):
    case = _case(tmp_path)
    after_rng = case.runtime._capture_rng()
    case.data.persist((after_rng,))
    calls = []

    def remote_prediction(task):
        calls.append(task["id"])
        raw_dir = tmp_path / "workers" / "root" / "raw"
        raw_dir.mkdir(parents=True)
        sidecar = raw_dir / "confidence.json"
        sidecar.write_bytes(b"changed")
        reference = snapshot_scores(
            StandardScores(names=["A"], plddt={"A": [0.9]}, complex_paths={}), case.data,
        )
        case.data.persist((reference,))
        return {
            **task, "raw_scores": asdict(reference), "worker_pid": 100,
            "constructor_count": 1, "prediction_wall_s": 0.0,
            "prediction_after_rng": asdict(after_rng), "job_raw_dir": str(raw_dir),
            "sidecars": [{
                "path": str(sidecar), "size_bytes": 8,
                "sha256": hashlib.sha256(b"original").hexdigest(),
            }],
        }

    try:
        with ThreadPoolExecutor(1) as pool:
            owner = _owner(case, pool, remote_prediction)
            with pytest.raises(ContractConflict, match="sidecar changed"):
                _run(owner)
            assert calls == [0]
            assert all(row[0] != "measure" for row in case.recorder.calls)
            # The paid output already has a durable completed native receipt.
            assert owner.trial._receipt(owner.root_job(protocol=PROTOCOL)).status == "completed"
            owner.shutdown()
    finally:
        case.close()


@pytest.mark.parametrize("damage", ["request_identity", "scoring_identity", "raw_scores", "score_sidecar"])
def test_foreign_score_evidence_cannot_cross_durable_member_boundary(tmp_path, damage):
    from astrolabe.design.refinement_worker import _blob, _sidecars

    case = _case(tmp_path)
    after_rng = case.runtime._capture_rng()
    case.data.persist((after_rng,))
    calls, rejected = [], []

    def remote_prediction(task):
        calls.append(task["id"])
        raw_dir = tmp_path / "workers" / "root" / "raw"
        raw_dir.mkdir(parents=True)
        reference = snapshot_scores(StandardScores(names=["A"], plddt={"A": [0.9]}, complex_paths={}), case.data)
        case.data.persist((reference,))
        result = _scored_result(case, task, {
            **task, "raw_scores": asdict(reference), "worker_pid": 100,
            "constructor_count": 1, "prediction_wall_s": 0.0,
            "prediction_after_rng": asdict(after_rng),
            "job_raw_dir": str(raw_dir), "sidecars": _sidecars(raw_dir),
        })
        if damage == "score_sidecar":
            (Path(result["score_raw_dir"]) / "derived.txt").write_text("changed")
        else:
            receipt = case.data.get(_blob(result["score_receipt"]), _SCORE_RECEIPT_CODEC)
            foreign = (snapshot_scores(StandardScores(names=["A"], plddt={"A": [0.1]}, complex_paths={}), case.data)
                       if damage == "raw_scores" else key_for("foreign-score"))
            changed = case.data.put(replace(receipt, **{damage: foreign}), _SCORE_RECEIPT_CODEC)
            case.data.persist((changed,))
            result["score_receipt"] = asdict(changed)
        return result

    try:
        with ThreadPoolExecutor(1) as pool:
            owner = _owner(case, pool, remote_prediction)

            def checked_result(job, raw):
                before = case.runtime._capture_rng()
                with pytest.raises(ContractConflict) as error:
                    owner._result(job, raw)
                assert case.runtime._capture_rng() == before
                rejected.append(job.id)
                raise error.value

            owner.trial.result_factory = checked_result
            with pytest.raises(ContractConflict):
                _run(owner)
            assert calls == rejected == [0]
            assert not any(item.key.named_round.endswith("/measurement")
                           for item in case.runtime.attempts())
            assert all(row[0] != "measure" for row in case.recorder.calls)
            owner.shutdown()
    finally:
        case.close()


def test_admitted_unissued_proposal_recovers_under_valid_runtime_binding(tmp_path):
    case = _case(tmp_path, max_evaluations=3)
    after_rng = case.runtime._capture_rng()
    case.data.persist((after_rng,))
    calls = []

    def remote_prediction(task):
        calls.append(task["id"])
        raw_dir = tmp_path / "workers" / str(task["id"]) / "raw"
        raw_dir.mkdir(parents=True)
        structure = raw_dir / "structure.cif"
        structure.write_text(task["sequence"])
        content = structure.read_bytes()
        scores = snapshot_scores(StandardScores(
            names=["A"], plddt={"A": [np.asarray([0.4, 0.8], dtype=np.float32)]},
            complex_paths={"A": structure},
        ), case.data)
        case.data.persist((scores,))
        return _scored_result(case, task, {
            **task, "raw_scores": asdict(scores), "worker_pid": 100,
            "constructor_count": 1, "prediction_wall_s": 0.0,
            "prediction_after_rng": asdict(after_rng), "job_raw_dir": str(raw_dir),
            "sidecars": [{
                "path": str(structure), "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }],
        })

    try:
        with ThreadPoolExecutor(2) as pool:
            owner = _owner(case, pool, remote_prediction)
            original = owner.trial._submit_task
            interrupted = False

            def interrupt_before_issue(pool, job, task):
                nonlocal interrupted
                if job.phase == "propose" and not interrupted:
                    interrupted = True
                    raise RuntimeError("injected before proposal issue")
                return original(pool, job, task)

            owner.trial._submit_task = interrupt_before_issue
            with pytest.raises(RuntimeError, match="injected before proposal issue"):
                _run(owner)
            owner.trial._submit_task = original
            assert case.runtime._work is None
            final = _run(owner)
            assert len(final.records) == 3
            assert sorted(calls) == [0, 1, 2]
            assert len([row for row in case.recorder.calls if row[0] == "propose"]) == 2
            assert not any(item.status == "unknown" for item in case.runtime.attempts())
            owner.shutdown()
    finally:
        case.close()


def test_raw_only_probe_restores_real_spawn_future_without_paid_replay(tmp_path):
    from omegaconf import OmegaConf
    from astrolabe.design.refinement_worker import execute
    from test_refinement_worker import _config, _spawn_initialize, _spawn_recovery_initialize

    case = _case(tmp_path, max_evaluations=1)

    def forbidden_parent_loss(*args):
        raise AssertionError("parent cannot calculate loss")

    case.spec = replace(case.spec, measure=forbidden_parent_loss)
    config = OmegaConf.to_container(OmegaConf.create(_config()), resolve=True)
    config["seqs"], config["types"] = {"A": "AC@shared"}, {"A": "protein"}
    config["stages"]["default"]["oligo"] = ["A"]
    args = (OmegaConf.to_yaml(OmegaConf.create(config)), str(case.data.root), ("A",),
            case.spec.translation.state, str(tmp_path / "workers"))
    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(1, mp_context=context, initializer=_spawn_initialize, initargs=args) as pool:
            owner = _owner(case, pool, execute)
            owner.bootstrap()
            job = owner.root_job(protocol=PROTOCOL)
            future = owner.trial._submit_task(owner.trial.executor.pool, job, owner._task(job))
            original = future.result(timeout=90)
        receipt = tmp_path / "workers" / "receipts" / f"{original['_request_hash']}.score.json"
        receipt.unlink()
        with ProcessPoolExecutor(1, mp_context=context, initializer=_spawn_recovery_initialize, initargs=args) as pool:
            recovered_owner = _owner(case, pool, execute)
            recovered_owner.bootstrap()
            recovered_owner.trial._restore(job)
            restored, = recovered_owner.trial.executor.one_pending
            recovered = restored.result(timeout=90)
            assert recovered["worker_pid"] == original["worker_pid"] != os.getpid()
            assert recovered["score_worker_pid"] != original["score_worker_pid"]
            assert recovered["raw_scores"] == original["raw_scores"]
            with case.runtime.bind("spawn-score-ingest", History(key_for(()), ())):
                recovered_owner.trial._observe(job, recovered)
                terminal = recovered_owner.trial._resolve(job, recovered)
            assert terminal.value.score == 0
            assert not any(row[0] in {"measure", "evaluate"} for row in case.recorder.calls)
            recovered_owner.shutdown()
        owner.shutdown()
    finally:
        case.close()
