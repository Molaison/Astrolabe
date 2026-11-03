"""The comparison gate must never turn changed science or failed work into speed."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/validation/runtime_workflow_benchmark.py"
SPEC = importlib.util.spec_from_file_location("workflow_benchmark", PATH)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def trial(mode, value="0.7", accepted="True", *, valid=True):
    return {"run": mode + value + accepted, "mode": mode, "valid_workload": valid,
            "seconds": 4, "allocated_gpu_seconds": 4, "completed": 1,
            "scientific_rows": {"(0, 17)": [{"step": "1", "accepted": accepted,
                                               "weighted_score": value, "A_sequence": "AAAA"}]}}


def test_direct_noise_does_not_permit_decision_changes():
    records = [trial("direct", "0.7"), trial("direct", "0.70001"),
               trial("resident_serial", "0.700005"), trial("resident_adaptive", "0.7", "False")]
    outcome = benchmark.comparison(records)
    assert not outcome["certified"]
    assert [(x["field"], x["kind"]) for x in outcome["differences"]] == [("accepted", "decision")]


def test_direct_baseline_instability_and_missing_samples_block_certification():
    unstable = benchmark.comparison([trial("direct"), trial("direct", accepted="False")])
    assert not unstable["certified"] and "unstable" in unstable["reason"]
    records = [trial("direct"), trial("direct"), trial("resident_adaptive", valid=False)]
    outcome = benchmark.report(records, 3, "frozen")
    assert not outcome["complete"] and outcome["performance_claim"] == "not certified"
    assert outcome["modes"]["resident_adaptive"]["workload_completions_per_second"] == 0
    assert outcome["modes"]["resident_adaptive"]["effective_trajectories_per_second"] is None


def test_precision_samples_and_seed_stay_in_protocol_identity(tmp_path):
    base = {"seed": 17, "repeats": 3, "evaluator": {"evaluator_terms": {
        "A": {"_type": "alphafold3", "num_diffusion_samples": 5, "precision": "bfloat16"}}}}
    for mode in benchmark.MODES:
        modified = benchmark.mode_config(base, mode, tmp_path / mode)
        assert benchmark.scientific_projection(modified) == base
    mutated = copy.deepcopy(base)
    mutated["evaluator"]["evaluator_terms"]["A"]["num_diffusion_samples"] = 1
    assert benchmark.fingerprint(benchmark.scientific_projection(mutated)) != benchmark.fingerprint(base)


def test_result_identity_and_nonfinite_values_are_not_silently_accepted():
    records = [trial("direct"), trial("direct"), trial("resident_adaptive")]
    records[-1]["scientific_rows"]["(1, 18)"] = records[-1]["scientific_rows"].pop("(0, 17)")
    assert benchmark.comparison(records)["differences"][0]["kind"] == "trajectory_identity"
    with pytest.raises(ValueError, match="nonfinite"):
        benchmark.comparison([trial("direct"), trial("direct", "nan")])


def test_legacy_mutator_factory_does_not_receive_new_execution_knobs(tmp_path):
    base = {"mutator": {"mutator_terms": {"MPNN": {"_type": "ProteinMPNN",
             "execution": "daemon", "execution_resources": {"output_bytes": 1024}, "temperature": .1}}}}
    config = benchmark.mode_config(base, "direct", tmp_path)
    assert config["mutator"]["mutator_terms"]["MPNN"] == {"_type": "ProteinMPNN", "temperature": .1}
    assert benchmark.scientific_projection(config) == benchmark.scientific_projection(base)


def test_missing_native_direct_baseline_is_not_relabelled_as_a_direct_run(tmp_path):
    with pytest.raises(ValueError, match="no legacy direct"):
        benchmark.mode_config({"evaluator": {"evaluator_terms": {"M": {"_type": "mol1"}}}}, "direct", tmp_path)


def test_numeric_gate_and_selection_changes_cannot_be_absorbed_as_noise():
    records = [trial("direct"), trial("direct")]
    for i, record in enumerate(records):
        row = next(iter(record["scientific_rows"].values()))[0]
        row["transition_consistency_gate_passed"] = str(i)
    assert benchmark.comparison(records)["reason"] == "direct scientific baseline is unstable"
    assert benchmark.decision_field("selected_sample_index")
    assert benchmark.decision_field("prefix_early_stop_triggered")
    assert benchmark.decision_field("decision.prefix_early_stop_triggered")


def test_repeated_or_wrong_scientific_identity_cannot_count_as_completed(tmp_path):
    def output(name, index, seed):
        run = tmp_path / "results" / (name + "_run")
        run.mkdir(parents=True)
        (run / "provenance.json").write_text(json.dumps({"state": "completed",
            "run": {"repeat_index": index, "seed": seed}}))
        (run / "scores.csv").write_text("step,accepted\n1,True\n")
    output("one", 0, 17)
    _, failures = benchmark.read_completed(tmp_path, {"(0, 18)"})
    assert failures[0]["missing"] == ["(0, 18)"]
    assert failures[0]["unexpected"] == ["(0, 17)"]
    output("duplicate", 0, 17)
    with pytest.raises(ValueError, match="duplicate scientific"):
        benchmark.read_completed(tmp_path)


def test_benchmark_preserves_per_repeat_seed_interpolation(tmp_path):
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "workload.yaml"
    config_path.write_text("task_name: interpolation\nrepeats: 3\nseed: 17\n"
                           "mutator:\n  MCMC:\n    seed: ${seed}\n")
    base = benchmark.compose(config_path, root / "config")
    assert base["mutator"]["MCMC"]["seed"] == "${seed}"
    identities = benchmark.planned_trajectories(base)
    assert identities == [{"repeat_index": i, "seed": 17 + i} for i in range(3)]
    for mode in benchmark.MODES:
        config = OmegaConf.create(benchmark.mode_config(base, mode, tmp_path / mode))
        for identity in identities:
            config.seed = identity["seed"]
            assert config.mutator.MCMC.seed == identity["seed"]


def test_default_seed_identity_matches_existing_deterministic_schedule():
    import hashlib
    rows = benchmark.planned_trajectories({"task_name": "cpu", "repeats": "2", "seed": None})
    assert rows == [{"repeat_index": i, "seed": (int(hashlib.sha256(f"cpu:{i}".encode()).hexdigest()[:8], 16) % 1_000_000) + 1} for i in range(2)]
    with pytest.raises(ValueError, match="explicit seed sequence"):
        benchmark.planned_trajectories({"repeats": 2, "seed": 2**32 - 1})


def test_slow_telemetry_does_not_extend_independent_workflow_timer(tmp_path, monkeypatch):
    import sys
    import time
    def slow_sample(_allocation):
        time.sleep(.6)
        return {"available": False}
    class TimerOnlyObserver:
        # Process-tree ownership has separate real-process tests. This test
        # isolates independent timing in executors without a matching /proc.
        def __init__(self, process):
            self.process = process
        def observe_until_exit(self, process, _deadline):
            code = process.wait()
            now = time.monotonic()
            return {"exit_code": code, "main_exit_monotonic": now,
                    "completion_monotonic": now, "release_confirmed": True, "stop_reason": None}
        def request_stop(self, _reason):
            self.process.terminate()
    monkeypatch.setattr(benchmark, "BenchmarkProcessObserver", TimerOnlyObserver)
    monkeypatch.setattr(benchmark, "gpu_sample", slow_sample)
    monkeypatch.setattr(benchmark, "process_sample", lambda *_: {"rss_bytes": None})
    started = time.monotonic()
    result = benchmark.run_once([sys.executable, "-c", "import time; time.sleep(.1)"],
                                tmp_path, tmp_path, "", started + 10, .5, 1024**2)
    wall = time.monotonic() - started
    assert result["exit_code"] == 0
    assert wall - result["seconds"] > .2


def test_source_identity_covers_untracked_execution_code(tmp_path):
    source = tmp_path / "astrolabe" / "new_worker.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n")
    before = benchmark.execution_environment(tmp_path)
    source.write_text("VALUE = 2\n")
    after = benchmark.execution_environment(tmp_path)
    assert before["source_manifest_sha256"] != after["source_manifest_sha256"]
    assert "astrolabe/new_worker.py" in after["source_manifest"]


def test_cpu_mutator_execution_controls_are_removed_for_old_factory(tmp_path):
    config = benchmark.mode_config({"mutator": {"mutator_terms": {
        "M": {"_type": "MCMC", "execution": "auto", "execution_resources": {"cpu_slots": 1}}}}},
        "direct", tmp_path)
    assert config["mutator"]["mutator_terms"]["M"] == {"_type": "MCMC"}
