"""CPU-only launcher contracts. Synthetic scheduler records; no GPU/job call."""
import ast
import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "benchmarks/runtime/gpu_slurm_launcher.py"
SPEC = importlib.util.spec_from_file_location("gpu_slurm_launcher", PATH)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
UUID = "GPU-12345678-1234-1234-1234-123456789abc"


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("CPU contracts must not start a process, Slurm job, container or GPU query")
    monkeypatch.setattr(launcher.subprocess, "run", blocked)
    monkeypatch.setattr(launcher.subprocess, "Popen", blocked)


@pytest.fixture
def environment():
    return {"SLURM_JOB_ID": "123456", "SLURM_STEP_ID": "0", "SLURM_JOB_NAME": "fresh-owned-control",
            "SLURM_STEP_GPUS": "4", "SLURM_JOB_GPUS": "4", "CUDA_VISIBLE_DEVICES": "0",
            "SLURM_CPUS_PER_TASK": "4", "SLURM_NTASKS": "1"}


@pytest.fixture
def job():
    # Slurm's slash/colon field names must not become part of NumTasks.
    return launcher.parse_scontrol(
        "JobId=123456 JobName=fresh-owned-control UserId=zpzeng(7054) GroupId=xcf(7010) "
        "Account=xulab QOS=normal JobState=RUNNING Partition=boron TimeLimit=00:15:00 "
        "NumNodes=1 NumCPUs=4 NumTasks=1 CPUs/Task=4 ReqB:S:C:T=0:0:*:* "
        "AllocTRES=cpu=4,mem=32G,node=1,gres/gpu=1 TresPerNode=gres/gpu:g4090:1")


def test_python36_grammar_and_no_shell_execution():
    tree = ast.parse(PATH.read_text(), feature_version=(3, 6))
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        assert not any(keyword.arg in ("capture_output", "text") for keyword in call.keywords)
        assert not any(keyword.arg == "shell" and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is False)
                       for keyword in call.keywords)
    assert "--jobid" not in PATH.read_text()
    assert "scancel" not in PATH.read_text()


def test_scheduler_assignment_cgroup_remap_and_scoped_query(environment, job):
    binding = launcher.allocated_binding(environment)
    assert binding["global_gpu_id"] == "4"
    assert binding["inherited_cuda_visible_devices"] == "0"  # Inherited, never guessed.
    launcher.verify_job(job, {"StepId": "123456.0", "State": "RUNNING"}, binding, "fresh-owned-control", 7054)
    command = launcher.gpu_query_argv(binding["global_gpu_id"])
    assert command[:3] == ["/usr/bin/nvidia-smi", "-i", "4"]
    assert command.count("-i") == 1
    result = launcher.parse_gpu_reply("4, " + UUID + ", synthetic RTX4090, driver-fixture, 0000:01:00.0, 24564, 0\n", binding)
    assert result["uuid"] == UUID


def test_complete_scheduler_uuid_survives_allocation_query_reply_roundtrip(environment):
    environment.update(SLURM_STEP_GPUS=UUID.upper(),
                       SLURM_JOB_GPUS=UUID, CUDA_VISIBLE_DEVICES=UUID)
    binding = launcher.allocated_binding(environment)
    assert binding["global_gpu_id"] == UUID
    assert launcher.gpu_query_argv(binding["global_gpu_id"])[2] == UUID
    gpu = launcher.parse_gpu_reply("5, " + UUID + ", name, driver, bus, 24564, 0", binding)
    assert gpu["uuid"] == UUID


@pytest.mark.parametrize("changes", [
    {"SLURM_JOB_ID": ""}, {"SLURM_STEP_ID": "batch"}, {"SLURM_STEPID": "1"},
    {"SLURM_JOB_GPUS": "5"}, {"SLURM_STEP_GPUS": "4,5"},
    {"SLURM_JOB_GPUS": "4-4", "SLURM_STEP_GPUS": "4-4"},
    {"SLURM_JOB_GPUS": "", "SLURM_STEP_GPUS": ""},
    {"CUDA_VISIBLE_DEVICES": ""}, {"CUDA_VISIBLE_DEVICES": "0,1"},
    {"CUDA_VISIBLE_DEVICES": "$(not-a-command)"}, {"SLURM_CPUS_PER_TASK": "8"}, {"SLURM_NTASKS": "2"},
])
def test_ambiguous_or_missing_binding_fails_before_any_query(environment, changes):
    environment.update(changes)
    with pytest.raises(ValueError):
        launcher.allocated_binding(environment)


@pytest.mark.parametrize("changes", [
    {"JobId": "31793"}, {"JobName": "existing-job"}, {"UserId": "other(9999)"},
    {"Account": "other"}, {"Partition": "iron"}, {"QOS": "other"},
    {"NumCPUs": "8"}, {"NumTasks": "2"}, {"NumNodes": "2"},
    {"TimeLimit": "UNLIMITED"}, {"TimeLimit": "00:16:00"},
    {"AllocTRES": "cpu=4,mem=64G,node=1,gres/gpu=1"},
    {"AllocTRES": "cpu=4,mem=32G,node=1,gres/gpu=2"},
    {"TresPerNode": "gres/gpu:g3090:1"}, {"JobState": "PENDING"},
])
def test_wrong_actual_scheduler_allocation_rejected(environment, job, changes):
    job.update(changes)
    with pytest.raises(ValueError):
        launcher.verify_job(job, {"StepId": "123456.0", "State": "RUNNING"},
                            launcher.allocated_binding(environment), "fresh-owned-control", 7054)


@pytest.mark.parametrize("reply", [
    "5, " + UUID + ", name, driver, bus, 24564, 0",
    "4, GPU-short, name, driver, bus, 24564, 0",
    ("4, " + UUID + ", name, driver, bus, 24564, 0\n") * 2,
    "", "4, " + UUID + ", name, driver, bus, N/A, 0",
])
def test_gpu_metadata_must_match_single_assigned_id(environment, reply):
    with pytest.raises(ValueError):
        launcher.parse_gpu_reply(reply, launcher.allocated_binding(environment))


def test_no_existing_job_submission(environment):
    with pytest.raises(ValueError):
        launcher.reject_existing_allocation(environment)
    with pytest.raises(ValueError):
        launcher.reject_existing_allocation({"CUDA_VISIBLE_DEVICES": "0"})
    launcher.reject_existing_allocation({})


def plan_fixture():
    directory = str(launcher.DELIVERY / "gpu-allocation-contract")
    return {"job_name": "astrolabe-gpu-20261104-fixture", "private": {name: directory + "/" + name
            for name in ("tmp", "var_tmp", "home", "cache", "apptainer_tmp", "apptainer_cache")},
            "readonly_files": [directory + "/frozen.json", "/readonly/weights.pt", "/readonly/input.pdb"],
            "parsed_input": directory + "/parsed_input"}


def test_exact_bounded_new_srun_argv_no_node_or_gpu_pin():
    argv = launcher.submission_argv(plan_fixture(), "/task/launch.json", "a" * 64)
    for required in ("--account=xulab", "--partition=boron", "--qos=normal", "--gres=gpu:g4090:1",
                     "--cpus-per-task=4", "--mem=32G", "--time=00:15:00", "--immediate=60",
                     "--nodes=1", "--ntasks=1"):
        assert argv.count(required) == 1
    assert not any(item.startswith(("--nodelist", "--exclusive", "--jobid", "--gpu-bind")) for item in argv)
    assert argv[-2:] == ["--launch-sha256", "a" * 64]


def test_container_gpu_visibility_readonly_sources_private_temp_and_cleanenv(environment):
    plan = plan_fixture()
    argv, env = launcher.container_argv(plan, {"uuid": UUID}, environment)
    assert "--nv" in argv and "--cleanenv" in argv and "--no-eval" in argv and "--no-home" in argv
    assert "--pid" not in argv and "--containall" not in argv  # Existing observer namespace preserved.
    assert env["CUDA_VISIBLE_DEVICES"] == UUID
    assert env["NVIDIA_VISIBLE_DEVICES"] == UUID
    assert all(env[key] == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
    assert env["GH_CUDA_PREINITIALIZE"] == "0"
    assert env["HIP_VISIBLE_DEVICES"] == env["ROCR_VISIBLE_DEVICES"] == ""
    assert env["TMPDIR"] == "/tmp" and env["HOME"].startswith(str(launcher.DELIVERY))
    binds = [argv[i + 1] for i, token in enumerate(argv) if token == "--bind"]
    assert plan["private"]["tmp"] + ":/tmp:rw" in binds
    assert str(launcher.ROOT) + ":" + str(launcher.ROOT) + ":ro" in binds
    assert "/readonly/weights.pt:/readonly/weights.pt:ro" in binds
    assert "/readonly/input.pdb:/readonly/input.pdb:ro" in binds
    cpu_argv, cpu_env = launcher.container_argv(plan)
    assert "--nv" not in cpu_argv and cpu_env["CUDA_VISIBLE_DEVICES"] == "" and cpu_env["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert cpu_env["HIP_VISIBLE_DEVICES"] == cpu_env["ROCR_VISIBLE_DEVICES"] == ""


def test_global_gpu_five_inherited_zero_binds_verified_uuid_before_cuda(environment):
    environment.update(SLURM_STEP_GPUS="5", SLURM_JOB_GPUS="5", CUDA_VISIBLE_DEVICES="0")
    binding = launcher.allocated_binding(environment)
    gpu = launcher.parse_gpu_reply("5, " + UUID + ", synthetic RTX4090, driver, bus, 24564, 0", binding)
    argv, env = launcher.container_argv(plan_fixture(), gpu, environment)
    assert binding["inherited_cuda_visible_devices"] == "0"
    assert env["CUDA_VISIBLE_DEVICES"] == env["NVIDIA_VISIBLE_DEVICES"] == UUID
    assert "CUDA_VISIBLE_DEVICES=0" not in argv


def test_untrusted_container_overrides_removed_without_breaking_slurm_client_libraries():
    env = {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/site/slurm/lib", "SRUN_JOBID": "old",
           "SLURM_TIME_FORMAT": "relative", "TZ": "Asia/Shanghai",
           "APPTAINER_BIND": "/old:/repo:rw", "APPTAINERENV_CUDA_VISIBLE_DEVICES": "all", "SINGULARITYENV_FOO": "bad"}
    private = plan_fixture()["private"]
    actual = launcher.clean_host_environment(env, private, container=False)
    assert actual["PATH"] == env["PATH"] and actual["LD_LIBRARY_PATH"] == env["LD_LIBRARY_PATH"]
    assert not any(key in actual for key in ("SRUN_JOBID", "APPTAINER_BIND", "APPTAINERENV_CUDA_VISIBLE_DEVICES", "SINGULARITYENV_FOO"))
    assert actual["APPTAINER_TMPDIR"].startswith(str(launcher.DELIVERY))
    assert actual["TZ"] == "UTC" and actual["SLURM_TIME_FORMAT"] == "standard"


def test_actual_allocation_clock_two_independent_representations():
    env = {"TZ": "UTC", "SLURM_JOB_START_TIME": "0", "SLURM_JOB_END_TIME": "900"}
    job = {"StartTime": "1970-01-01T00:00:00", "EndTime": "1970-01-01T00:15:00"}
    assert launcher.allocation_clock(env, job) == (0, 900)
    assert launcher.allocation_clock({"TZ": "UTC"}, job) == (0, 900)
    with pytest.raises(ValueError):
        launcher.allocation_clock({**env, "SLURM_JOB_END_TIME": "901"}, job)
    with pytest.raises(ValueError):
        launcher.allocation_clock({}, {})


def test_terminal_accounting_is_own_job_and_actual_cost_not_client_wall():
    line = "123456|COMPLETED|0:0|2026-11-04T10:00:00|2026-11-04T10:04:00|240|cpu=4,mem=32G,node=1,gres/gpu=1\n"
    actual = launcher.accounting_summary(line + line.replace("123456|", "123456.0|"), "123456")
    assert actual["actual_gpu_hours"] == 240 / 3600 and actual["within_gpu_budget"]
    for bad in (line.replace("123456|", "999|"), line.replace("COMPLETED", "RUNNING"), line.replace("2026-11-04T10:04:00", "Unknown")):
        with pytest.raises(ValueError):
            launcher.accounting_summary(bad, "123456")
    overrun = launcher.accounting_summary(line.replace("|240|", "|901|"), "123456")
    assert not overrun["within_gpu_budget"]


@pytest.mark.parametrize("change", [
    {"start": "not-a-date"}, {"end": "2026-11-04T09:59:00"},
    {"start": "2026-02-30T10:00:00"}, {"code": "unknown"},
    {"allocation": "cpu=4,mem=32G,node=1,gres/gpu=1,gres/gpu:g4090=2"},
    {"allocation": "cpu=4,mem=32G,node=1,gres/gpu=1,gres/gpu:g3090=1"},
])
def test_terminal_accounting_rejects_invalid_dates_exit_and_contradictory_gpu_counts(change):
    fields = {"start": "2026-11-04T10:00:00", "end": "2026-11-04T10:04:00", "code": "0:0",
              "allocation": "cpu=4,mem=32G,node=1,gres/gpu=1,gres/gpu:g4090=1"}
    fields.update(change)
    line = "123456|COMPLETED|{code}|{start}|{end}|240|{allocation}".format(**fields)
    with pytest.raises(ValueError):
        launcher.accounting_summary(line, "123456")


def test_accounting_preserves_elapsed_wall_difference_and_gates_both_budgets():
    line = "123456|COMPLETED|0:0|2026-11-04T10:00:00|2026-11-04T10:04:00|200|gres/gpu:g4090=1"
    actual = launcher.accounting_summary(line, "123456")
    assert actual["actual_wall_seconds"] == 240 and actual["actual_elapsed_seconds"] == 200
    assert actual["actual_gpu_hours"] == 200 / 3600
    assert actual["allocation_wall_gpu_hours"] == 240 / 3600
    assert actual["within_gpu_budget"]
    wall_overrun = launcher.accounting_summary(line.replace("10:04:00", "10:16:00"), "123456")
    assert not wall_overrun["within_gpu_budget"]


def test_bind_path_delimiters_are_not_interpreted():
    for value in ("/task/evil,other", "/task/evil:/other", "/task/new\nline"):
        with pytest.raises(ValueError):
            launcher.safe_path(value)
    # A metacharacter is literal in argv; --no-eval also fences container env expansion.
    assert "$" in str(launcher.safe_path("/task/literal$token"))
