import json

from scripts.validation.runtime_trace_report import summarize


def test_composite_time_unknown_residency_and_no_telemetry_are_not_gpu_evidence(tmp_path):
    path = tmp_path / "trace.jsonl"
    events = [
        {"event": "model_loaded", "backend": "private-A", "pid": 12,
         "lifecycle": {"worker_epoch": "epoch", "load_count": None, "gpu_resident": None}},
        {"event": "request_complete", "backend": "private-A", "cohort": "model-version",
         "shape_key": "4-residues", "phase": "composite", "cold_start": True,
         "outcome": "completed", "duration": 2,
         "lifecycle": {"worker_epoch": "epoch", "load_count": 1, "gpu_resident": False}},
        {"event": "compute_end", "backend": "private-A", "phase": "composite", "duration": 1.5},
        {"event": "request_complete", "backend": "private-A", "cohort": "model-version",
         "shape_key": "4-residues", "phase": "composite", "cold_start": False,
         "outcome": "completed", "duration": 1},
        {"event": "resource_sample", "time": 1, "devices": {}},
        {"event": "resource_sample", "time": 2, "devices": {}},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    report = summarize(path)
    assert report["known_model_load_count"] == 1
    assert report["time_weighted_sampled_gpu_activity_percent"] is None
    assert {row["phase"] for row in report["phase_observations"]} == {"composite"}
    assert {row["temperature"] for row in report["request_observations"]} == {"cold", "steady"}
    assert not report["sampled_low_activity_seconds_by_control_context"]


def test_bad_or_excessive_trace_is_visible_instead_of_fabricated(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text('broken\n' + json.dumps({"event": "request_complete", "backend": "A"}) + '\n')
    report = summarize(path, maximum_groups=0)
    assert report["malformed_lines"] == 1 and report["group_limit_drops"] == 1
    assert not summarize(tmp_path / "missing")["available"]
