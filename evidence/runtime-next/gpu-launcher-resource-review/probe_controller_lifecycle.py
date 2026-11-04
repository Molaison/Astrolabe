"""Caller-level launcher outcomes with controlled children; no process may spawn."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

repo, output = (Path(value).resolve() for value in sys.argv[1:3])
output.mkdir(exist_ok=False)
spec = importlib.util.spec_from_file_location("launcher_control_review", repo / "benchmarks/runtime/gpu_slurm_launcher.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
results = []


class Child:
    def __init__(self, stubborn=False, exited=False):
        self.stubborn, self.returncode, self.actions = stubborn, (0 if exited else None), []

    def poll(self):
        self.actions.append("poll")
        return self.returncode

    def wait(self, timeout):
        self.actions.append(["wait", timeout])
        if self.stubborn and self.returncode is None:
            raise subprocess.TimeoutExpired("controlled own child", timeout)
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self):
        self.actions.append("terminate")
        if not self.stubborn:
            self.returncode = -15

    def kill(self):
        self.actions.append("kill")
        self.returncode = -9


for case in ("successful_submit", "delayed_accounting", "own_child_timeout"):
    directory = output / case
    directory.mkdir()
    (directory / "allocation.json").write_text(json.dumps({"binding": {"job_id": "123456"}}))
    plan = {"directory": str(directory), "frozen": str(directory / "frozen.json"), "frozen_sha256": "b" * 64,
            "job_name": "review-unique-job", "private": {"apptainer_tmp": str(directory / "tmp"),
                                                        "apptainer_cache": str(directory / "cache")}}
    args = SimpleNamespace(launch=directory / "launch.json", launch_sha256="a" * 64)
    own, unrelated, commands, spawned = Child(stubborn=case == "own_child_timeout"), Child(), [], []

    def popen(argv, **kwargs):
        assert argv[0] == launcher.SRUN and "--gres=gpu:g4090:1" in argv
        assert "--time=00:15:00" in argv and "--immediate=60" in argv
        assert not any(str(a).startswith(("--jobid", "--nodelist", "--exclusive")) for a in argv)
        spawned.append(list(argv))
        return own

    def metadata(argv, _directory, _name, **kwargs):
        assert argv[0] == launcher.SACCT
        assert [a for a in argv if a.startswith("--jobs=")] == ["--jobs=123456"]
        commands.append(list(argv))
        state = "RUNNING" if case == "delayed_accounting" else ("CANCELLED by 7054" if case == "own_child_timeout" else "COMPLETED")
        code = "0:15" if case == "own_child_timeout" else "0:0"
        end = "Unknown" if case == "delayed_accounting" else "2026-11-04T10:04:00"
        return {"returncode": 0, "stdout": "123456|%s|%s|2026-11-04T10:00:00|%s|240|cpu=4,mem=32G,node=1,gres/gpu=1,gres/gpu:g4090=1\n" % (state, code, end), "stderr": ""}

    with patch.dict(os.environ, {}, clear=True), \
         patch.object(launcher, "read_launch", return_value=plan), \
         patch.object(launcher, "verify_frozen", return_value={}), \
         patch.object(launcher, "command_record", side_effect=metadata), \
         patch.object(launcher.subprocess, "Popen", side_effect=popen), \
         patch.object(launcher.subprocess, "run", side_effect=AssertionError("No real command permitted")):
        code = launcher.submit(args)
        raw = json.loads((directory / "submission.json").read_text())
        assert len(spawned) == 1 and len(commands) == 1 and not unrelated.actions
        if case == "successful_submit":
            assert code == 0 and raw["status"] == "COMPLETE" and "terminate" not in own.actions
            preserved = (directory / "submission.json").read_bytes()
            try:
                launcher.submit(args)
            except (ValueError, FileExistsError):
                pass
            else:
                raise AssertionError("Second submission must be rejected")
            assert len(spawned) == 1 and preserved == (directory / "submission.json").read_bytes()
        elif case == "delayed_accounting":
            assert code == 2 and raw["status"] == "PARTIAL_ACCOUNTING_OR_BUDGET"
            assert "terminal_accounting" not in raw
        else:
            assert code == 1 and raw["status"] == "FAILED"
            assert own.actions == [["wait", 1020], "poll", "terminate", ["wait", 5], "kill", ["wait", 5], "poll"]
    results.append({"case": case, "pass": True, "status": raw["status"], "exit_code": code,
                    "spawned": spawned, "metadata_commands": commands, "own_child_actions": own.actions,
                    "unrelated_child_actions": unrelated.actions})

# Validate the whole inside caller's ownership failure boundary. All scheduler
# reads are synthetic; an attempted GPU query/container launch is a test failure.
directory = output / "inside_foreign_owner"
directory.mkdir()
plan = {"directory": str(directory), "frozen": str(directory / "frozen.json"),
        "frozen_sha256": "b" * 64, "job_name": "review-unique-job"}
args = SimpleNamespace(launch=directory / "launch.json", launch_sha256="a" * 64)
env = {"SLURM_JOB_ID": "123456", "SLURM_STEP_ID": "0", "SLURM_JOB_NAME": "review-unique-job",
       "SLURM_JOB_GPUS": "5", "SLURM_STEP_GPUS": "5", "CUDA_VISIBLE_DEVICES": "0",
       "SLURM_CPUS_PER_TASK": "4", "SLURM_NTASKS": "1", "TZ": "UTC"}
commands = []


def foreign_metadata(argv, _directory, _name, **kwargs):
    assert argv[0] == launcher.SCONTROL, "Ownership failure must prevent GPU query"
    assert argv[-1] in ("123456", "123456.0")
    commands.append(argv)
    text = "JobId=123456 JobName=review-unique-job UserId=other(9999) Account=xulab QOS=normal Partition=boron JobState=RUNNING"
    if "step" in argv:
        text = "StepId=123456.0 State=RUNNING"
    return {"returncode": 0, "stdout": text, "stderr": ""}


with patch.dict(os.environ, env, clear=True), \
     patch.object(launcher, "read_launch", return_value=plan), \
     patch.object(launcher, "verify_frozen", return_value={}), \
     patch.object(launcher.os, "getuid", return_value=7054), \
     patch.object(launcher, "command_record", side_effect=foreign_metadata), \
     patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("No real child permitted")), \
     patch.object(launcher.subprocess, "run", side_effect=AssertionError("No real command permitted")):
    code = launcher.inside(args)
    raw = json.loads((directory / "allocation.json").read_text())
    assert code == 1 and raw["status"] == "FAILED" and not raw["container_started"]
    assert "owned by the current user" in raw["error"]["message"]
    assert len(commands) == 4  # before/after reads all name this same job/step.
results.append({"case": "inside_foreign_owner", "pass": True, "status": raw["status"],
                "container_started": raw["container_started"], "metadata_commands": commands})

exited, unrelated = Child(exited=True), Child()
launcher.terminate_child(exited)
assert exited.actions == ["poll"] and unrelated.actions == []
results.append({"case": "already_exited_child", "pass": True, "own_child_actions": exited.actions,
                "unrelated_child_actions": unrelated.actions})
result = {"scope": "Controlled child/metadata boundaries only; no process, allocation, container or GPU was started",
          "passed": len(results), "cases": results}
(output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"passed": len(results), "scope": result["scope"]}))
