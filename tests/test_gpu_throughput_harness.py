from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO_ROOT / "scripts" / "validation" / "model_gpu_throughput.py"


def _load_harness():
    name = "model_gpu_throughput_harness_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HARNESS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint_dir(tmp_path: Path) -> Path:
    harness = _load_harness()
    directory = tmp_path / "weights"
    directory.mkdir()
    (directory / harness.CHECKPOINT_NAME).write_bytes(b"real checkpoint bytes")
    return directory


def test_cli_defaults_requests_and_checkpoint_resolution(tmp_path: Path) -> None:
    harness = _load_harness()
    weights = _checkpoint_dir(tmp_path)
    source = tmp_path / "input.cif"
    source.write_text("data_test\n#\n", encoding="utf-8")
    out = tmp_path / "out"
    args = harness.parse_args(
        [
            "--out-dir",
            str(out),
            "--weights",
            str(weights),
            "--input",
            str(source),
        ]
    )
    assert args.requests == 4
    assert args.seed == 17
    assert args.chain == "A"
    assert args.weights == weights / harness.CHECKPOINT_NAME


def test_cli_rejects_one_request_and_nonempty_output(
    tmp_path: Path,
    capsys,
) -> None:
    harness = _load_harness()
    weights = _checkpoint_dir(tmp_path)
    source = tmp_path / "input.cif"
    source.write_text("data_test\n#\n", encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        harness.parse_args(
            [
                "--out-dir",
                str(out),
                "--weights",
                str(weights),
                "--input",
                str(source),
                "--requests",
                "1",
            ]
        )
    assert "--requests must be at least 2" in capsys.readouterr().err

    out.mkdir()
    (out / "existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        harness.parse_args(
            [
                "--out-dir",
                str(out),
                "--weights",
                str(weights),
                "--input",
                str(source),
            ]
        )
    assert "must be absent or empty" in capsys.readouterr().err


def test_timing_summary_uses_synchronized_wall_boundaries() -> None:
    harness = _load_harness()
    events: list[str] = []
    ticks = iter((10.0, 10.25))

    def synchronize() -> None:
        events.append("sync")

    def clock() -> float:
        events.append("clock")
        return next(ticks)

    value, elapsed = harness._measure(
        lambda: events.append("call") or "result",
        synchronize=synchronize,
        clock=clock,
    )
    assert value == "result"
    assert elapsed == pytest.approx(0.25)
    assert events == ["sync", "clock", "call", "sync", "clock"]


def test_timing_aggregation_reports_all_warm_and_default_three_warm_rate() -> None:
    harness = _load_harness()
    summary = harness.aggregate_timings(2.0, [1.0, 0.5, 0.25, 0.25])
    assert summary.complete_cold_start_s == pytest.approx(4.0)
    assert summary.warm_total_s == pytest.approx(1.0)
    assert summary.warm_requests_per_s == pytest.approx(3.0)
    assert summary.first_three_warm_total_s == pytest.approx(1.0)
    assert summary.first_three_warm_requests_per_s == pytest.approx(3.0)


def test_timing_aggregation_supports_more_requests_without_hiding_them() -> None:
    harness = _load_harness()
    summary = harness.aggregate_timings(0.5, [2.0, 1.0, 1.0, 1.0, 1.0])
    assert summary.warm_total_s == pytest.approx(4.0)
    assert summary.warm_requests_per_s == pytest.approx(1.0)
    assert summary.first_three_warm_total_s == pytest.approx(3.0)
    assert summary.first_three_warm_requests_per_s == pytest.approx(1.0)


def test_real_loader_wrapper_delegates_and_restores_original() -> None:
    harness = _load_harness()
    observed = []

    def original(path, *args, **kwargs):
        observed.append((path, args, kwargs))
        return {"real": "loaded object"}

    class FakeTorch:
        load = staticmethod(original)

    reads: list[str] = []
    with harness.counted_torch_load(FakeTorch, reads) as traced:
        loaded = FakeTorch.load("weights.pt", map_location="cuda")
        assert FakeTorch.load.__name__ == "loader"
    assert loaded == {"real": "loaded object"}
    assert reads == ["weights.pt"]
    assert traced is reads
    assert observed == [("weights.pt", (), {"map_location": "cuda"})]
    assert FakeTorch.load is original


def test_exact_comparison_preserves_dtype_shape_nan_and_order() -> None:
    harness = _load_harness()
    expected = {
        "generated_sequences": np.asarray([[1, 2, 3]], dtype=np.int32),
        "sampling_probs": np.asarray([1.0, np.nan], dtype=np.float32),
        "seed": 17,
        "chain_letters": ("A",),
    }
    actual = {
        "generated_sequences": np.asarray([[1, 2, 3]], dtype=np.int32),
        "sampling_probs": np.asarray([1.0, np.nan], dtype=np.float32),
        "seed": 17,
        "chain_letters": ("A",),
    }
    equal = harness.compare_outputs(expected, actual)
    assert equal["equal"] is True
    assert equal["differences"] == []

    actual["sampling_probs"] = np.asarray([1.0, np.nan], dtype=np.float64)
    dtype = harness.compare_outputs(expected, actual)
    assert dtype["equal"] is False
    assert {"path": "$.sampling_probs", "reason": "dtype"} in dtype["differences"]

    actual["sampling_probs"] = np.asarray([1.0, np.nan], dtype=np.float32)
    actual["generated_sequences"] = np.asarray([[1, 2]], dtype=np.int32)
    shape = harness.compare_outputs(expected, actual)
    assert {"path": "$.generated_sequences", "reason": "shape"} in shape["differences"]


def test_dtype_aware_comparison_handles_strings_complex_and_object_payloads() -> None:
    harness = _load_harness()

    def object_array(payload):
        result = np.empty(1, dtype=object)
        result[0] = payload
        return result

    expected_payload = {
        "encoded_residues": np.asarray(["A1", "A2"], dtype="<U3"),
        "complex_values": np.asarray(
            [1.0 + 2.0j, np.nan + 3.0j],
            dtype=np.complex128,
        ),
        "structured": object_array(
            {
                "labels": np.asarray(["B1", "B2"], dtype="<U3"),
                "indices": (1, 2),
            }
        ),
    }
    actual_payload = {
        "encoded_residues": expected_payload["encoded_residues"].copy(),
        "complex_values": expected_payload["complex_values"].copy(),
        "structured": object_array(
            {
                "labels": np.asarray(["B1", "B2"], dtype="<U3"),
                "indices": (1, 2),
            }
        ),
    }

    equal = harness.compare_outputs(expected_payload, actual_payload)
    assert equal["equal"] is True
    assert equal["differences"] == []
    assert equal["expected_sha256"] == equal["actual_sha256"]

    actual_payload["encoded_residues"][1] = "U2"
    string_difference = harness.compare_outputs(expected_payload, actual_payload)
    assert {
        "path": "$.encoded_residues",
        "reason": "value",
    } in string_difference["differences"]

    actual_payload["encoded_residues"] = expected_payload[
        "encoded_residues"
    ].copy()
    actual_payload["structured"][0]["indices"] = (1, 3)
    object_difference = harness.compare_outputs(expected_payload, actual_payload)
    assert {
        "path": "$.structured",
        "reason": "value",
    } in object_difference["differences"]


def test_comparison_rejects_unsupported_output_instead_of_dropping_it() -> None:
    harness = _load_harness()
    with pytest.raises(harness.HarnessError, match="Unsupported ProteinMPNN output"):
        harness.compare_outputs({"opaque": object()}, {"opaque": object()})


def test_checkpoint_trace_requires_exact_real_path(tmp_path: Path) -> None:
    harness = _load_harness()
    checkpoint = tmp_path / harness.CHECKPOINT_NAME
    checkpoint.write_bytes(b"checkpoint")
    harness._assert_exact_trace([str(checkpoint)], checkpoint, lane="direct")
    with pytest.raises(harness.HarnessError, match="exactly"):
        harness._assert_exact_trace([str(checkpoint), str(checkpoint)], checkpoint, lane="direct")
    with pytest.raises(harness.HarnessError, match="exactly"):
        harness._assert_exact_trace([], checkpoint, lane="daemon")


def test_atomic_report_and_source_identity_are_concrete(tmp_path: Path) -> None:
    harness = _load_harness()
    path = tmp_path / "throughput.json"
    harness._atomic_json(path, {"schema": harness.SCHEMA, "verified": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema": harness.SCHEMA,
        "verified": True,
    }
    assert not path.with_name(f".{path.name}.{os.getpid()}.tmp").exists()
    identity = harness._source_identity()
    assert identity["scripts/validation/model_gpu_throughput.py"] == harness._sha256(
        HARNESS_PATH
    )
    assert identity["astrolabe/model_execution.py"] == harness._sha256(
        REPO_ROOT / "astrolabe" / "model_execution.py"
    )


def test_main_does_not_open_torch_or_models_when_runner_is_injected(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    harness = _load_harness()
    weights = _checkpoint_dir(tmp_path)
    source = tmp_path / "input.cif"
    source.write_text("data_test\n#\n", encoding="utf-8")
    called = []

    def run(args):
        called.append(args)
        return {"verified": True}

    monkeypatch.setattr(harness, "_run", run)
    assert harness.main(
        [
            "--out-dir",
            str(tmp_path / "out"),
            "--weights",
            str(weights),
            "--input",
            str(source),
        ]
    ) == 0
    assert called[0].requests == 4
    assert capsys.readouterr().out.strip().endswith("throughput.json")
