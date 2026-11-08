"""CPU-only acceptance of resident evaluator workers and durable raw receipts."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

import run
from astrolabe import app
from astrolabe.core import global_config
from astrolabe.design import refinement_worker as worker
from astrolabe.design.legacy import LegacyContext
from astrolabe.design.native_scores import restore_scores
from astrolabe.design.native_refinement_runtime import _SCORE_RECEIPT_CODEC
from astrolabe.design.runtime import ContractConflict, OutcomeUnknown
from astrolabe.design.store import DataStore, design_codec
from astrolabe.design.types import Assignment, Blob, Key, Polymer
from astrolabe.evaluators.registry import EvaluatorRegistry
from astrolabe.evaluators.utils import StandardScores
from astrolabe.model_execution import restore_random_state
from astrolabe.losses.base import LossEngine


STATE = "worker-test-state"
ASSEMBLIES = ("AB", "C")


def _config():
    return OmegaConf.to_yaml(OmegaConf.create({
        "task_name": "worker_cpu", "seed": 71, "out_dir": "/unused",
        "raw_model_results_dir": "/unused", "seqs": {"A": "AC", "B": "GG", "C": "+GG"},
        "types": {"A": "protein", "B": "protein", "C": "protein"},
        "homologies": {}, "named_files": {}, "debug": False,
        "exclude_AA": "", "initializer": {"mode": "random"},
        "enable_pyrosetta": False, "skip_complex_poses": True,
        "per_residue_scores": None, "post_eval_hooks": None,
        "structure_start": None, "obabel_path": None, "predocker": {"predocker_terms": {}},
        "mutator": {"mutator_terms": {"unused": {"_type": "never_construct"}},
                    "selector_expression": None},
        "evaluator": {"evaluator_terms": {"fold": {"_type": "worker_fake"}}},
        "stages": {"default": {"oligo": list(ASSEMBLIES),
                               "weighted_score_expression": "0",
                               "mutators": ["unused"], "mutation_num": 1,
                               "evaluator": "fold"}},
        "loss": {"loss_terms": {}, "expression_terms": {}},
    }))


class _Evaluator:
    def __init__(self, output_dir, calls):
        self.output_dir = Path(output_dir)
        self.calls = calls
        self.fail = False
        self.shutdown_calls = 0
        # Constructor randomness must not affect per-request prediction RNG.
        random.random()
        np.random.random(13)

    def evaluate(self, inputs, runtime_config):
        sequences = {name: [chain.sequence for chain in value.chains]
                     for name, value in inputs.items()}
        draw = [random.random(), float(np.random.random())]
        self.calls.append({"sequences": sequences, "step": global_config.step,
                           "seeds": {name: list(value.rng_seeds) for name, value in inputs.items()},
                           "output": str(self.output_dir), "draw": draw})
        if self.fail:
            raise RuntimeError("injected native prediction failure")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        content = json.dumps(sequences, sort_keys=True)
        (self.output_dir / "confidence.json").write_text(content)
        structure = Path(global_config.out_dir) / "structures" / f"step_{global_config.step}.cif"
        structure.write_text(content)
        # Exercise restoration even when an evaluator mutates legacy state.
        global_config.protomers["A"].sequence  # real native projection above
        next(iter(global_config.segments.values())).sequence = "YY"
        scores = StandardScores(
            names=list(inputs),
            plddt={name: [np.asarray(draw, dtype=np.float32)] for name in inputs},
            complex_paths={next(iter(inputs)): structure},
        )
        scores.worker_test_extension = {"draw": draw, "sequences": sequences}
        return scores

    def shutdown(self):
        self.shutdown_calls += 1


def _install(monkeypatch, calls):
    monkeypatch.setattr(app, "_load_run_module", lambda: run)

    def factory(term_conf, *, alias, raw_model_results_dir, oligo_names, evaluator_conf):
        return _Evaluator(Path(raw_model_results_dir) / "fake", calls)

    monkeypatch.setitem(EvaluatorRegistry._registry, "worker_fake", factory)

    def forbidden(*args, **kwargs):
        raise AssertionError("worker must not construct mutators or replay initialization")

    monkeypatch.setattr(run, "build_mutators", forbidden)
    monkeypatch.setattr(run, "apply_initializer", forbidden)
    monkeypatch.setattr(run, "load_pipeline_components", forbidden)


@pytest.fixture(autouse=True)
def _isolate_worker(monkeypatch):
    monkeypatch.setattr(worker, "_WORKER", None)
    global_config.reset()
    yield
    if worker._WORKER is not None:
        worker._WORKER.shutdown()
        worker._WORKER.finalizer.cancel()
        worker.atexit.unregister(worker._WORKER.shutdown)
    global_config.reset()


@pytest.fixture
def session(tmp_path, monkeypatch):
    calls = []
    _install(monkeypatch, calls)
    args = (_config(), str(tmp_path / "data"), ASSEMBLIES, STATE, str(tmp_path / "worker"))
    worker.initialize(*args)
    return worker._WORKER, calls, args


def _task(session, *, request="request-one", replacements=None, seed=71, step=3):
    owner, _calls, _args = session
    translation = owner.translation
    candidate = translation.candidate
    assignments = tuple(
        Assignment(assignment.variable, replace(assignment.value, tokens=tuple(replacements[assignment.variable])))
        for assignment in candidate.assignments
        if replacements and assignment.variable in replacements
    )
    if assignments:
        candidate = translation.system.edit(candidate, assignments, data=owner.data)
    blob = owner.data.put(candidate, design_codec)
    owner.data.persist((blob,))
    return {"id": 0, "kind": "evaluate", "generation": 1, "parent_id": 9,
            "seed": seed, "sequence": "opaque-parent-label", "step": step,
            "_request_hash": request, "candidate_blob": asdict(blob),
            "candidate_identity": candidate.identity.digest,
            "system_identity": candidate.system.digest,
            "scoring_identity": "a" * 64,
            "loss_order": list(owner.loss.loss_specs),
            "job": {"identity": {"digest": "parent-job-digest"}, "codec": "parent-job/v1"}}


def _fresh_data(owner):
    data = DataStore(owner.data.root)
    for codec in owner.codecs.values():
        data.register(codec)
    return data


def _replace_worker(owner, args, role):
    owner.shutdown()
    owner.finalizer.cancel()
    worker.atexit.unregister(owner.shutdown)
    worker._WORKER = None
    worker.initialize(*args, role=role)
    return worker._WORKER


def test_split_roles_reuse_raw_without_score_model_or_prediction_replay(session, monkeypatch):
    owner, calls, args = session
    task = _task(session)
    raw = worker.execute(task)
    (owner.receipts / "request-one.score.json").unlink()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Score must not construct or evaluate a model")

    monkeypatch.setattr(run, "construct_evaluators", forbidden)
    monkeypatch.setattr(run, "perform_evaluation", forbidden)
    scorer = _replace_worker(owner, args, "score")
    assert scorer.constructor_count == 0
    assert scorer.evaluators == {}
    scored = worker.execute(task)
    assert scored["raw_scores"] == raw["raw_scores"]
    assert scored["score_worker_pid"] == os.getpid()
    assert len(calls) == 1
    assert worker.ready() == json.loads((scorer.session_root / "ready.json").read_text())
    with pytest.raises(OutcomeUnknown):
        worker.execute({**task, "_request_hash": "missing-raw"})
    Path(raw["sidecars"][0]["path"]).write_text("damaged")
    with pytest.raises(OutcomeUnknown):
        worker.execute(task)
    assert len(calls) == 1


def test_prediction_role_persists_only_raw(session, monkeypatch):
    owner, calls, args = session
    task = _task(session)
    predictor = _replace_worker(owner, args, "prediction")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("prediction role must not calculate Score")

    monkeypatch.setattr(predictor, "_score", forbidden)
    raw = worker.execute(task)
    assert "score_receipt" not in raw
    assert (predictor.receipts / "request-one.json").is_file()
    assert not (predictor.receipts / "request-one.score.json").exists()
    assert worker.execute(task) == raw
    assert len(calls) == 1


@pytest.mark.parametrize("dynamic_params", [False, True])
def test_score_ready_records_real_pyrosetta_init_or_dynamic_skip(
    session, monkeypatch, dynamic_params,
):
    owner, _calls, args = session
    config = OmegaConf.create(args[0])
    config.enable_pyrosetta = True
    config.skip_complex_poses = False
    config.molfile_to_params = "/unused-for-test"
    config.loss.loss_terms = {
        "buns": {"_type": "interface_buried_unsatisfied_hbonds", "ignore_surface_res": True},
    }
    config.stages.default.weighted_score_expression = "buns"
    initialized, calls = [False], []

    def init_runtime(*_args, **_kwargs):
        calls.append("init")
        initialized[0] = True

    monkeypatch.setattr(run, "_needs_molfile_to_params", lambda _args: dynamic_params)
    monkeypatch.setattr(run, "setup_extra_residues", lambda **_kwargs: None)
    monkeypatch.setattr(run, "_init_pyrosetta_runtime", init_runtime)
    from astrolabe.losses import interaction
    monkeypatch.setattr(
        interaction, "_make_interface_buns_filter",
        lambda *, ignore_surface_res: calls.append(("buns", ignore_surface_res)),
    )
    monkeypatch.setitem(worker.sys.modules, "pyrosetta", SimpleNamespace(
        is_initialized=lambda: initialized[0],
    ))
    scorer = _replace_worker(owner, (OmegaConf.to_yaml(config), *args[1:]), "score")
    ready = worker.ready()
    assert ready["pid"] == scorer.pid
    assert ready["role"] == "score"
    assert ready["started_unix"] <= ready["finished_unix"]
    assert ready["initialization_seconds"] >= 0
    assert ready["pyrosetta_initialized"] is (not dynamic_params)
    assert calls == ([] if dynamic_params else ["init", ("buns", True)])
    assert ready["warmed_score_resources"] == ([] if dynamic_params else [
        {"filter": "BuriedUnsatHbonds", "ignore_surface_res": True},
    ])


def test_full_candidates_fixed_seed_and_request_outputs_are_isolated(session):
    owner, calls, _ = session
    mutable = [assignment.variable for assignment in owner.translation.candidate.assignments
               if isinstance(assignment.value, Polymer) and assignment.value.tokens == ("A", "C")][0]
    first = _task(session, replacements={mutable: "DE"})
    second = _task(session, request="request-two")
    r1, r2 = worker.execute(first), worker.execute(second)
    assert calls[0]["sequences"] == {"AB": ["DE", "GG"], "C": ["GG"]}
    assert calls[1]["sequences"] == {"AB": ["AC", "GG"], "C": ["GG"]}
    assert calls[0]["draw"] == calls[1]["draw"]
    assert calls[0]["step"] == calls[1]["step"] == 3
    assert calls[0]["seeds"] == calls[1]["seeds"] == {"AB": [71], "C": [71]}
    assert r1["worker_pid"] == r2["worker_pid"] == os.getpid()
    assert r1["constructor_count"] == r2["constructor_count"] == 1
    assert r1["job_raw_dir"] != r2["job_raw_dir"]
    assert Path(calls[0]["output"]).is_relative_to(r1["job_raw_dir"])
    assert Path(calls[1]["output"]).is_relative_to(r2["job_raw_dir"])
    assert global_config.protomers["A"].sequence == "AC"
    for request, result in ((first, r1), (second, r2)):
        assert result["request"] == request
        assert all(result[name] == value for name, value in request.items())
        assert result["prediction_wall_s"] >= 0
        for item in result["sidecars"]:
            content = Path(item["path"]).read_bytes()
            assert item["size_bytes"] == len(content)
            assert item["sha256"] == hashlib.sha256(content).hexdigest()


def test_raw_scores_and_prediction_rng_recover_from_disk(session, tmp_path):
    owner, calls, _ = session
    task = _task(session)
    result = worker.execute(task)
    data = _fresh_data(owner)
    scores = restore_scores(worker._blob(result["raw_scores"]), data, tmp_path / "restored")
    assert scores.worker_test_extension["draw"] == calls[0]["draw"]
    assert scores.worker_test_extension["sequences"] == calls[0]["sequences"]
    expected = json.dumps(calls[0]["sequences"], sort_keys=True)
    original = Path(owner.root / "requests" / task["_request_hash"] / "structures" / "step_3.cif")
    original.unlink()
    assert scores.complex_paths["AB"].read_text() == expected
    restore_random_state(data.get(worker._blob(result["prediction_after_rng"]), design_codec))
    expected_next = [random.random(), float(np.random.random())]
    random.seed(71)
    np.random.seed(71)
    random.random()
    np.random.random()
    assert [random.random(), float(np.random.random())] == expected_next
    owner.data._memory.clear()
    assert worker.execute(task) == result
    assert len(calls) == 1


@pytest.mark.parametrize("field,value", [
    ("id", 100), ("kind", "evaluate"), ("generation", 5), ("parent_id", 77),
    ("seed", 72), ("step", 9), ("sequence", "other-label"),
    ("candidate_identity", "other-candidate"), ("system_identity", "other-system"),
    ("job", {"identity": {"digest": "changed"}, "codec": "parent-job/v1"}),
])
def test_completed_hash_requires_entire_request_identity(session, field, value):
    _owner, calls, _ = session
    task = _task(session)
    worker.execute(task)
    changed = {**task, field: value}
    if field == "kind":
        changed["new-parent-metadata"] = True
    with pytest.raises(ContractConflict):
        worker.execute(changed)
    assert len(calls) == 1


@pytest.mark.parametrize("phase", ["evaluation", "snapshot", "persist", "commit"])
def test_unknown_never_replays_and_preserves_issued(session, monkeypatch, phase):
    owner, calls, _ = session
    task = _task(session)

    def fail(*args, **kwargs):
        raise RuntimeError(f"injected {phase} failure")

    if phase == "evaluation":
        owner.evaluator.fail = True
    elif phase == "snapshot":
        monkeypatch.setattr(worker, "snapshot_scores", fail)
    elif phase == "persist":
        monkeypatch.setattr(owner.data, "persist", fail)
    else:
        monkeypatch.setattr(worker, "_complete", fail)
    with pytest.raises(OutcomeUnknown):
        worker.execute(task)
    assert (owner.receipts / "request-one.issued.json").exists()
    assert not (owner.receipts / "request-one.json").exists()
    owner.evaluator.fail = False
    with pytest.raises(OutcomeUnknown):
        worker.execute(task)
    with pytest.raises(ContractConflict):
        worker.execute({**task, "sequence": "changed"})
    assert len(calls) == 1
    assert global_config.protomers["A"].sequence == "AC"


@pytest.mark.parametrize("damage", ["truncated", "missing-score", "sidecar", "missing-rng"])
def test_incomplete_completed_receipt_is_unknown(session, damage):
    owner, calls, _ = session
    task = _task(session)
    result = worker.execute(task)
    receipt = owner.receipts / "request-one.json"
    if damage == "truncated":
        receipt.write_text('{"request":')
    elif damage == "missing-score":
        owner.data._path_for(worker._blob(result["raw_scores"])).unlink()
    elif damage == "missing-rng":
        owner.data._path_for(worker._blob(result["prediction_after_rng"])).unlink()
    else:
        Path(result["sidecars"][0]["path"]).write_text("modified")
    with pytest.raises(OutcomeUnknown):
        worker.execute(task)
    assert len(calls) == 1


@pytest.mark.parametrize("record", ["{", "{}", '{"request":null}'])
def test_incomplete_issued_receipt_cannot_replay(session, record):
    owner, calls, _ = session
    task = _task(session)
    (owner.receipts / "request-one.issued.json").write_text(record)
    with pytest.raises((OutcomeUnknown, ContractConflict)):
        worker.execute(task)
    assert not calls


@pytest.mark.parametrize("mode", ["random", "rfd3", "structure"])
def test_initializer_is_never_replayed_and_cleanup_is_idempotent(tmp_path, monkeypatch, mode):
    calls = []
    _install(monkeypatch, calls)
    config = OmegaConf.create(_config())
    config.initializer.mode = mode
    worker.initialize(OmegaConf.to_yaml(config), str(tmp_path / "data"), ASSEMBLIES,
                      STATE, str(tmp_path / "worker"))
    owner = worker._WORKER
    task = _task((owner, calls, None))
    worker.execute(task)
    assert owner.translation.state == STATE
    owner.shutdown()
    owner.finalizer()
    owner.shutdown()
    assert owner.evaluator.shutdown_calls == 1


@pytest.mark.parametrize("field", ["candidate_identity", "system_identity"])
def test_candidate_identity_mismatch_is_rejected_before_issue(session, field):
    owner, calls, _ = session
    task = _task(session)
    task[field] = "not-the-persisted-candidate"
    with pytest.raises(ContractConflict):
        worker.execute(task)
    assert not list(owner.receipts.iterdir())
    assert not calls


def test_candidate_from_another_state_is_rejected_before_issue(session):
    owner, calls, _ = session
    other = LegacyContext().capture_candidate(assemblies=ASSEMBLIES, state="different", data=owner.data)
    candidate = owner.data.put(other.candidate, design_codec)
    owner.data.persist((candidate,))
    task = _task(session)
    task.update(candidate_blob=asdict(candidate), candidate_identity=other.candidate.identity.digest,
                system_identity=other.candidate.system.digest)
    with pytest.raises(ValueError, match="different system"):
        worker.execute(task)
    assert not calls
    assert not list(owner.receipts.iterdir())


def _spawn_initialize(*args):
    # Monkeypatch inside the child, rather than relying on inherited globals.
    patches = pytest.MonkeyPatch()
    _install(patches, [])
    worker.initialize(*args)


def _spawn_recovery_initialize(*args):
    _spawn_initialize(*args)
    worker._WORKER.evaluator.fail = True


def test_spawn_resident_evaluator_and_restart_receipt_recovery(session):
    owner, _calls, args = session
    task = _task(session)
    second = _task(session, request="request-two", step=3)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context,
                             initializer=_spawn_initialize, initargs=args) as pool:
        first = pool.submit(worker.execute, task).result(timeout=60)
        another = pool.submit(worker.execute, second).result(timeout=60)
    assert first["worker_pid"] == another["worker_pid"] != os.getpid()
    assert first["constructor_count"] == another["constructor_count"] == 1
    assert first["job_raw_dir"] != another["job_raw_dir"]
    # A fresh process returns the original receipt without prediction replay.
    with ProcessPoolExecutor(max_workers=1, mp_context=context,
                             initializer=_spawn_initialize, initargs=args) as pool:
        recovered = pool.submit(worker.execute, task).result(timeout=60)
    assert recovered == first


def _spawn_scoring_initialize(args, barrier, parent_pid):
    _spawn_initialize(*args)
    owner = worker._WORKER
    original = owner.loss.compute_loss

    def compute(scores, runtime_config):
        assert os.getpid() != parent_pid
        request = Path(global_config.out_dir).name
        assert (owner.receipts / f"{request}.json").is_file()
        assert not (owner.receipts / f"{request}.score.json").exists()
        started = time.monotonic()
        barrier.wait(timeout=30)
        value = sum(index * index for index in range(200000))
        result = original(scores, runtime_config)
        (global_config.raw_model_results_dir / "derived.txt").write_text(str(value))
        scores.score_test_extension = {
            "pid": os.getpid(), "started": started, "finished": time.monotonic(),
            "draw": [random.random(), float(np.random.random())],
            "active": global_config.protomers["A"].mutated_sequence,
            "term_order": list(owner.loss.loss_specs),
            "context": runtime_config,
        }
        return result

    owner.loss.compute_loss = compute


def test_spawn_cpu_scores_overlap_and_raw_only_recovery_never_predicts(session):
    owner, _calls, args = session
    first = _task(session)
    second = _task(session, request="request-two")
    config = OmegaConf.to_container(OmegaConf.create(args[0]), resolve=True)
    config["loss"]["loss_terms"] = {
        "left": {"_type": "plddt", "oligomer_specific": ["AB"]},
        "right": {"_type": "plddt", "oligomer_specific": ["C"]},
    }
    config["stages"]["default"]["weighted_score_expression"] = "2 * left - right"
    args = (OmegaConf.to_yaml(OmegaConf.create(config)), *args[1:])
    from astrolabe.design.configured_refinement import snapshot_score_context
    score_context = {
        "state_conflict_diagnostics": OmegaConf.create({"enabled": False, "terms": ["left"]}),
        "score_catalog": [{"source": "mutator.dplm", "name": "entropy_norm", "entity": "AB"}],
        "score_queries": {"entropy": {"source": "mutator.dplm", "name": "entropy_norm", "snapshot": "producer:DPLM"}},
        "strict_score_contributions": [{"source": "mutator.dplm", "name": "entropy_norm", "value": .7,
                                         "snapshot": "producer-snapshot", "input_snapshot": "accepted-snapshot",
                                         "subject": "proposal", "relation": "proposal_on_reference"}],
        "score_snapshots": {"last_accepted": "accepted-snapshot", "producer:DPLM": "producer-snapshot"},
    }
    context_blob = snapshot_score_context(score_context, owner.data)
    frozen_context = restore_scores(context_blob, owner.data, owner.root / "context-view")
    assert frozen_context.configured_runtime_config["state_conflict_diagnostics"] == {
        "enabled": False, "terms": ["left"],
    }
    for task in (first, second):
        task["loss_order"] = ["right", "left"]
        task["score_context"] = asdict(context_blob)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    with ProcessPoolExecutor(max_workers=2, mp_context=context,
                             initializer=_spawn_scoring_initialize,
                             initargs=(args, barrier, os.getpid())) as pool:
        futures = [pool.submit(worker.execute, task) for task in (first, second)]
        receipts = [future.result(timeout=90) for future in futures]
    score_receipts = [_fresh_data(owner).get(worker._blob(item["score_receipt"]), _SCORE_RECEIPT_CODEC)
                      for item in receipts]
    views = [restore_scores(item.measured_scores, _fresh_data(owner), owner.root / f"view-{index}")
             for index, item in enumerate(score_receipts)]
    assert receipts[0]["score_worker_pid"] != receipts[1]["score_worker_pid"]
    assert all(item["score_worker_pid"] != os.getpid() for item in receipts)
    raw_scores = restore_scores(score_receipts[0].raw_scores, _fresh_data(owner), owner.root / "serial-input")
    engine = LossEngine(config["loss"], validate_config=False)
    engine.setup("2 * left - right", None, None, validate_config=False)
    engine.loss_specs = {name: engine.loss_specs[name] for name in first["loss_order"]}
    engine.loss_names = first["loss_order"]
    from astrolabe.design.configured_refinement import measure_configured_scores
    serial = measure_configured_scores(engine, raw_scores, {}, ASSEMBLIES)
    assert score_receipts[0].measurement.score == score_receipts[1].measurement.score == serial.score
    assert score_receipts[0].measurement.metrics["loss"] == score_receipts[1].measurement.metrics["loss"] == serial.metrics["loss"]
    assert all(view.score_test_extension["term_order"] == ["right", "left"] for view in views)
    assert all(view.score_test_extension["context"] == score_context for view in views)
    assert score_receipts[0].after_rng == score_receipts[1].after_rng
    assert views[0].score_test_extension["draw"] == views[1].score_test_extension["draw"]
    intervals = [view.score_test_extension for view in views]
    assert max(row["started"] for row in intervals) < min(row["finished"] for row in intervals)
    assert [view.score_test_extension["active"] for view in views] == ["AC", "AC"]
    assert all(not (Path(item["job_raw_dir"]) / "derived.txt").exists() for item in receipts)
    assert all((Path(item["score_raw_dir"]) / "derived.txt").is_file() for item in receipts)
    # Removing only the CPU receipt preserves the paid raw prediction. A fresh
    # worker with an evaluator that fails must still complete Score recovery.
    (owner.receipts / "request-one.score.json").unlink()
    owner.evaluator.fail = True
    owner.loss = engine
    recovered = worker.execute(first)
    assert not owner.evaluator.calls
    recovered_score = _fresh_data(owner).get(worker._blob(recovered["score_receipt"]), _SCORE_RECEIPT_CODEC)
    assert recovered_score.raw_scores == score_receipts[0].raw_scores
    assert recovered_score.measurement.score == score_receipts[0].measurement.score
