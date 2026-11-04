"""Test evidence gates, not performance. Synthetic rows are never benchmarks."""
import copy
import importlib.util
import json
from pathlib import Path
import socket

import pytest

PATH = Path(__file__).resolve().parents[1] / "benchmarks/runtime/validate.py"
SPEC = importlib.util.spec_from_file_location("runtime_validation", PATH)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


def scientific_rows():
    return {identity: [{"step": str(step), "weighted_score": "0.75",
                       "accepted": "True" if step == 0 else "False", "pA_sequence": "AAAA"}
                      for step in range(3)] for identity in validation.IDENTITIES}


def records():
    result = []
    for group in range(validation.GROUPS):
        for mode in validation.MODES:
            for variant in validation.VARIANTS:
                result.append({"run": len(result) + 1, "repetition": group, "variant": variant,
                               "mode": mode, "valid_workload": True, "completed": 3,
                               "seconds": 3.0 if variant == "baseline" else 2.0,
                               "scientific_rows": scientific_rows(), "structure_artifacts": {},
                               "work_counts": {"evaluation_calls": 9}})
    return result


def write_trace(path, evaluations=9):
    from astrolabe.runtime.contracts import identity_key
    backend = identity_key({"backend": "astrolabe.evaluators.dummy.DummyEvaluator"})
    events = [{"event": "stage", "backend": backend, "phase": phase}
              for _ in range(evaluations) for phase in validation.PHASES]
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def test_all_fixed_work_is_required_before_any_speed_ratio():
    complete = validation.comparison_report(records())
    assert complete["status"] == "DONE"
    assert all(pair["median_paired_speed_ratio"] == 1.5 for pair in complete["same_mode_pairs"])
    assert complete["best_direct_selection"] == "lower median of the two direct variants"
    assert all(pair["direct_variant"] == "candidate" for pair in complete["candidate_vs_best_direct"])
    partial = validation.comparison_report(records()[:-1])
    assert partial["status"] == "PARTIAL"
    assert all(pair["median_paired_speed_ratio"] is None for pair in partial["same_mode_pairs"])
    assert all(group["effective_trajectories_per_second"] is None for group in partial["groups"].values())
    assert not partial["candidate_vs_best_direct"]


def test_candidate_direct_cannot_expand_scientific_tolerance():
    rows = records()
    for record in rows:
        if record["variant"] == "candidate":
            for trajectory in record["scientific_rows"].values():
                trajectory[0]["weighted_score"] = "0.750001"
    result = validation.comparison_report(rows)
    assert result["complete"]
    assert not result["correctness"]["certified"]
    assert all(item["absolute"] == 0 for item in result["correctness"]["tolerances"])
    assert all(pair["median_paired_speed_ratio"] is None for pair in result["same_mode_pairs"])


@pytest.mark.parametrize("kind", ["failed_fast", "duplicate", "nonfinite", "wrong_sequence"])
def test_invalid_work_cannot_be_a_speedup(kind):
    rows = records()
    if kind == "failed_fast":
        rows[-1].update(valid_workload=False, seconds=.001)
    elif kind == "duplicate":
        rows[-1] = copy.deepcopy(rows[-2])
    else:
        row = rows[-1]["scientific_rows"]["(0, 17)"][0]
        row["weighted_score" if kind == "nonfinite" else "pA_sequence"] = "nan" if kind == "nonfinite" else "WWWW"
    result = validation.comparison_report(rows)
    assert result["status"] == "PARTIAL"
    assert all(pair["paired_speed_ratios"] is None for pair in result["same_mode_pairs"])


def test_trace_and_scientific_rows_prove_different_work_counts(tmp_path):
    trace = tmp_path / "runtime.jsonl"
    rows = scientific_rows()
    missing = validation.work_counts(rows, trace)
    assert not missing["certified"] and missing["evaluation_calls"] is None
    write_trace(trace)
    valid = validation.work_counts(rows, trace)
    assert valid["certified"] and valid["evaluation_calls"] == 9
    rows["(0, 17)"].pop()
    assert not validation.work_counts(rows, trace)["certified"]
    write_trace(trace, evaluations=10)
    extra = validation.work_counts(scientific_rows(), trace)
    assert not extra["certified"] and extra["evaluation_calls"] == 10
    trace.write_text("not JSON\n")
    assert validation.work_counts(scientific_rows(), trace)["evaluation_calls"] is None


def test_environment_denial_is_reported_without_a_fallback(monkeypatch):
    monkeypatch.setattr(validation, "process_namespace_supported", lambda: False)
    def denied(*args, **kwargs):
        raise PermissionError(1, "local test: UNIX sockets unavailable")
    monkeypatch.setattr(socket, "socket", denied)
    result = validation.preflight()
    assert result["status"] == "BLOCKED"
    assert "matching_process_namespace" in result["blocking_reasons"]
    assert "af_unix" in result["blocking_reasons"]
    assert result["af_unix_error"]["errno"] == 1


def test_required_suite_skip_reason_is_preserved_and_fails_gate(tmp_path):
    junit = tmp_path / "junit.xml"
    junit.write_text('<testsuites><testsuite><testcase classname="runtime" name="ownership">'
                     '<skipped message="executor PID API and /proc namespaces disagree"/>'
                     '</testcase></testsuite></testsuites>')
    report = validation.junit_result(junit, 0)
    assert report["status"] == "PARTIAL"
    assert report["skips"] == [{"test": "runtime::ownership", "reason": "executor PID API and /proc namespaces disagree"}]
    assert "tests/test_candidate_batch_policy.py" in validation.REQUIRED_TESTS


def test_torch_contract_summary_does_not_require_or_claim_a_benchmark(tmp_path):
    for phase in ("preflight", "regressions"):
        directory = tmp_path / phase
        directory.mkdir()
        (directory / "report.json").write_text('{"status":"DONE"}')
    assert validation.main(["summary", "--torch-contracts", "--out", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report == {"status": "DONE", "gates": {"preflight": "DONE", "regressions": "DONE"}}
    text = (tmp_path / "summary.md").read_text()
    assert "CPU Torch and structural contract validation" in text
    assert "No pretrained inference or performance benchmark is certified" in text
    assert "frozen baseline" not in text.lower()


def test_existing_evidence_is_never_overwritten(tmp_path):
    marker = tmp_path / "already-measured.json"
    marker.write_text("original")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        validation.fresh_directory(tmp_path)
    assert marker.read_text() == "original"


def test_real_w0_composition_retains_frozen_scientific_protocol():
    base = validation.benchmark.compose(validation.ROOT / "scripts/validation/fixtures/runtime_cpu.yaml",
                                        validation.ROOT / "config")
    assert validation.benchmark.fingerprint(validation.benchmark.scientific_projection(base)) == validation.PROTOCOL
    identities = validation.benchmark.planned_trajectories(base)
    assert identities == [{"repeat_index": i, "seed": 17 + i} for i in range(3)]


def test_plan_preserves_work_and_alternates_complete_revision_pairs(tmp_path, monkeypatch):
    baseline, out = tmp_path / "baseline", tmp_path / "plan"
    out.mkdir()
    for name in validation.MEASUREMENT_FILES:
        path = baseline / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((validation.ROOT / name).read_bytes())
    monkeypatch.setattr(validation, "source_identity", lambda *args, **kwargs: {"git_commit": validation.BASELINE})
    plan = validation.build_plan(baseline, validation.ROOT, out)
    assert plan["requested_trials"] == 30
    assert plan["walltime_budget_seconds"] == 900
    assert plan["max_output_bytes"] == 260_000_000
    assert plan["trajectories_per_trial"] == 3 and plan["evaluations_per_trial"] == 9
    for group in range(5):
        trials = [row for row in plan["runs"] if row["repetition"] == group]
        assert len(trials) == 6
        for first, second in zip(trials[::2], trials[1::2]):
            assert first["mode"] == second["mode"]
            assert [first["variant"], second["variant"]] == (["baseline", "candidate"] if group % 2 == 0 else ["candidate", "baseline"])
        assert len({row["mode"] for row in trials}) == 3
    for variant in validation.VARIANTS:
        for mode in validation.MODES:
            assert sum((row["variant"], row["mode"]) == (variant, mode) for row in plan["runs"]) == 5
