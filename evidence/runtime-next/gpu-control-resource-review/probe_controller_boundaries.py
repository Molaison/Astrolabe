"""Independent CPU controller-boundary probes; spawning/model/GPU work is forbidden."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

repo, output = map(lambda value: Path(value).resolve(), sys.argv[1:3])
output.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(repo))
spec = importlib.util.spec_from_file_location("gpu_control_resource_review", repo / "benchmarks/runtime/gpu_mutator_control.py")
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
from astrolabe.runtime import telemetry
from scripts.validation import runtime_workflow_benchmark as measurement_module

protocol = json.loads(control.DEFAULT_PROTOCOL.read_text())
control.validate_protocol(protocol)
uuid = "GPU-12345678-1234-1234-1234-123456789abc"
reports = []

for case in ("positive", "nonzero_exit", "unknown_release", "forced_cleanup",
             "measurement_exception", "overall_deadline", "pre_output_budget", "post_output_budget"):
    directory = output / case
    directory.mkdir()
    local_protocol = copy.deepcopy(protocol)
    if case in ("pre_output_budget", "post_output_budget"):
        # This is a small controller-only negative-boundary fixture. The real
        # frozen protocol is not changed or executed; load_frozen is mocked.
        local_protocol["budget"]["max_output_bytes"] = 4096
    frozen = {"output": str(directory), "protocol": local_protocol, "plan": control.trial_plan(local_protocol)}
    frozen_path = directory / "frozen.json"
    frozen_path.write_text("{}\n")
    if case == "pre_output_budget":
        (directory / "preexisting.txt").write_bytes(b"x" * 4096)
    clock, calls = [100.0], []
    args = SimpleNamespace(frozen=frozen_path, frozen_sha256="controller-fixture-only", allocated_gpu_uuid=uuid)

    def fake_run_once(command, trial_dir, cwd, allocation, deadline, sample_interval, cap, repository):
        entered = clock[0]
        calls.append({"argv": list(command), "allocation": allocation, "deadline": deadline,
                      "clock_before": entered, "cap": cap, "sample_interval": sample_interval,
                      "cwd": str(cwd), "repository": str(repository)})
        assert allocation == uuid and cwd == repo and repository == repo
        assert isinstance(command, list) and command[2] == "_trial"
        assert command[command.index("--frozen") + 1] == str(frozen_path)
        assert command[command.index("--allocated-gpu-uuid") + 1] == uuid
        assert 0 < deadline - entered <= protocol["budget"]["trial_seconds"]
        assert 0 < cap <= local_protocol["budget"]["max_output_bytes"]
        assert sample_interval == .5
        clock[0] += 5.0
        if case == "measurement_exception":
            raise RuntimeError("controlled measurement failure; no process was launched")
        (trial_dir / "synthetic-bytes.txt").write_bytes(b"x" * (8192 if case == "post_output_budget" else 128))
        observation = {"release_confirmed": True, "exit_code": 0, "cleanup_attempted": False,
                       "observed_tree_empty": True, "remaining_identities": []}
        reason = None
        if case == "nonzero_exit":
            observation["exit_code"] = 1
        if case == "unknown_release":
            observation.update(release_confirmed=False, observed_tree_empty=False, remaining_identities=[[123, 456]])
        if case == "forced_cleanup":
            observation.update(release_confirmed=False, cleanup_attempted=True)
            reason = "output_budget_exhausted"
        if case == "overall_deadline":
            clock[0] = 1000.0
        return {"seconds": 2.0, "measurement_envelope_seconds": 5.0, "process_release": observation,
                "stop_reason": reason, "peak_process_tree_rss_bytes": None, "peak_device_vram_bytes": None}

    def fake_aggregate(_path, execution):
        # Only test controller boundaries here. Real aggregate/science is
        # independently exercised by the actual CPU tensor fixture tests.
        return {"status": "PASS" if execution["status"] == "COMPLETE" else "INVALID",
                "completed_design_requests": 0,
                "pairs": [{"pass": True} for _ in range(len(execution["trials"]) // 2)],
                "scope": "SYNTHETIC controller-only, zero scientific evaluations"}

    env = {"SLURM_JOB_ID": "synthetic-unit-only", "CUDA_VISIBLE_DEVICES": "0",
           "NVIDIA_VISIBLE_DEVICES": uuid}
    with patch.dict(os.environ, env, clear=True), \
         patch.object(control, "load_frozen", return_value=frozen), \
         patch.object(control, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 0.0)), \
         patch.object(control, "aggregate", side_effect=fake_aggregate), \
         patch.object(telemetry, "process_namespace_supported", return_value=True), \
         patch.object(measurement_module, "run_once", side_effect=fake_run_once), \
         patch.object(control.subprocess, "Popen", side_effect=AssertionError("Any real spawn is forbidden in this probe")):
        code = control.run(args)

    execution = json.loads((directory / "execution.json").read_text())
    expected_calls = 10 if case == "positive" else (0 if case == "pre_output_budget" else 1)
    assert len(calls) == expected_calls, (case, len(calls))
    assert code == (0 if case == "positive" else 1), (case, code)
    if case == "positive":
        assert [call["argv"][call["argv"].index("--index") + 1] for call in calls] == [str(i) for i in range(10)]
        assert all(row["wall_seconds"] == 2.0 and row["measurement"]["measurement_envelope_seconds"] == 5.0
                   for row in execution["trials"])
        assert all(a["cap"] >= b["cap"] for a, b in zip(calls, calls[1:]))
    reports.append({"case": case, "exit_code": code, "run_once_calls": len(calls),
                    "status": execution["status"], "error": execution.get("error", {}).get("message"),
                    "calls": calls, "pass": True})

result = {"scope": "Synthetic run_once/aggregate/namespace boundary only; real subprocess spawning is prohibited; no GPU/worker/model/Slurm execution",
          "source": str(repo), "cases": reports, "passed": len(reports), "scientific_evaluations": 0}
(output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"passed": len(reports), "scope": result["scope"]}))
