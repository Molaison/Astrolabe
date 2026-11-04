"""Independent CPU-only boundary probe. No Slurm/container/GPU process exists."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
from unittest.mock import patch

repo, output = (Path(value).resolve() for value in sys.argv[1:3])
assert not output.exists()
spec = importlib.util.spec_from_file_location("resource_launch_review", repo / "benchmarks/runtime/gpu_slurm_launcher.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
uuid = "GPU-12345678-1234-1234-1234-123456789abc"
environment = {"SLURM_JOB_ID": "123456", "SLURM_STEP_ID": "0", "SLURM_JOB_NAME": "fresh-review",
               "SLURM_STEP_GPUS": "5", "SLURM_JOB_GPUS": "5", "CUDA_VISIBLE_DEVICES": "0",
               "SLURM_CPUS_PER_TASK": "4", "SLURM_NTASKS": "1"}
account = "123456|COMPLETED|0:0|2026-11-04T10:00:00|2026-11-04T10:04:00|240|cpu=4,mem=32G,node=1,gres/gpu=1,gres/gpu:g4090=1\n"
results = []


def check(name, expected_accept, call):
    try:
        value = call()
    except (ValueError, TypeError) as exc:
        row = {"case": name, "accepted": False, "error": type(exc).__name__ + ": " + str(exc)}
    else:
        row = {"case": name, "accepted": True, "value": value}
    row["expected_accept"] = expected_accept
    row["pass"] = row["accepted"] == expected_accept
    results.append(row)


def supported_uuid_roundtrip():
    env = {**environment, "SLURM_JOB_GPUS": uuid, "SLURM_STEP_GPUS": uuid, "CUDA_VISIBLE_DEVICES": uuid}
    binding = launcher.allocated_binding(env)
    # The returned token must remain valid for the next real caller's validation.
    return {"binding": binding, "query_argv": launcher.gpu_query_argv(binding["global_gpu_id"])}


def canonical_utc_environment():
    private = {"apptainer_tmp": "/new-task/private/tmp", "apptainer_cache": "/new-task/private/cache"}
    actual = launcher.clean_host_environment({"SLURM_TIME_FORMAT": "relative", "TZ": "Asia/Shanghai"}, private, container=False)
    if actual.get("TZ") != "UTC" or actual.get("SLURM_TIME_FORMAT") != "standard":
        raise ValueError("Launcher did not fix both timezone and timestamp format: " + repr(actual))
    return {key: actual[key] for key in ("TZ", "SLURM_TIME_FORMAT")}


with patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("No process may be launched")), \
     patch.object(launcher.subprocess, "run", side_effect=AssertionError("No command may be launched")):
    check("numeric_global_id_five_inherited_zero", True,
          lambda: launcher.gpu_query_argv(launcher.allocated_binding(environment)["global_gpu_id"]))
    check("supported_complete_uuid_roundtrip", True, supported_uuid_roundtrip)
    check("explicit_utc_standard_format", True, canonical_utc_environment)
    check("valid_terminal_accounting", True, lambda: launcher.accounting_summary(account, "123456"))
    check("invalid_start_end_dates", False,
          lambda: launcher.accounting_summary(account.replace("2026-11-04T10:00:00", "not-a-date")
                                                    .replace("2026-11-04T10:04:00", "also-not-a-date"), "123456"))
    check("end_before_start", False,
          lambda: launcher.accounting_summary(account.replace("2026-11-04T10:04:00", "2026-11-04T09:59:00"), "123456"))
    check("contradictory_typed_gpu_count", False,
          lambda: launcher.accounting_summary(account.replace("gres/gpu:g4090=1", "gres/gpu:g4090=2"), "123456"))

result = {"scope": "Synthetic scheduler metadata only; subprocesses forbidden; no allocation, container or GPU operation",
          "cases": results, "passed": sum(row["pass"] for row in results), "total": len(results)}
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
assert all(row["pass"] for row in results), "Launcher metadata boundaries remain inconsistent"
