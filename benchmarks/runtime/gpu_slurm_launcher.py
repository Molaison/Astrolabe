#!/usr/bin/env python3
"""Task-specific, Python-3.6-compatible single-job launcher. No shell evaluation.

prepare: real CPU-only container/mount check, no Slurm request or GPU query.
submit: one fresh bounded srun; never attaches to or retries an existing job.
_inside: query only that job/step and its single scheduler-assigned GPU ID.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/xcf26/zpzeng/practice_arena/astrolabe_runtime_20261102_implementation")
DELIVERY = BASE / "cloud-20261104"
TEST_ENV = BASE / "test-env"
APPTAINER_PREFIX = Path("/xcf26/zpzeng/.gh_container/envs/_gh_apptainer")
APPTAINER = APPTAINER_PREFIX / "bin/apptainer"
SIF = Path("/xcf26/zpzeng/.gh_container/images/GuidedHallucination_af3_licensed_a13_20260722.sif")
SIF_BYTES = 6418354176
SIF_SHA256 = "242aae3bb97c5c7ea374dab5d7462e04863d9d48ba2ee00cc57e6c0a57a538e3"
HOST_PYTHON = "/usr/bin/python3"
CONTAINER_PYTHON = "/alphafold3_venv/bin/python"
SRUN = "/usr/bin/srun"
SCONTROL = "/usr/bin/scontrol"
SACCT = "/usr/bin/sacct"
NVIDIA_SMI = "/usr/bin/nvidia-smi"
GPU_UUID = r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
SLURM_FIELDS = ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_JOB_NAME", "SLURM_JOB_ACCOUNT",
                "SLURM_JOB_PARTITION", "SLURM_JOB_QOS", "SLURM_JOB_NODELIST", "SLURMD_NODENAME",
                "SLURM_STEP_ID", "SLURM_STEPID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
                "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_MEM_PER_NODE", "SLURM_NTASKS",
                "SLURM_JOB_START_TIME", "SLURM_JOB_END_TIME")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def safe_path(value):
    value = str(value)
    require(not any(character in value for character in ("\0", "\r", "\n", ",", ":")),
            "ambiguous Apptainer bind path")
    return Path(value).resolve()


def beneath(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def stat_identity(path):
    info = Path(path).stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns, "device": info.st_dev, "inode": info.st_ino}


def reject_existing_allocation(environment):
    keys = ("SLURM_JOB_ID", "SLURM_JOBID", "SLURM_STEP_ID", "SLURM_STEPID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS")
    require(not any(environment.get(key) for key in keys), "submission must be outside every existing allocation")
    require(not environment.get("CUDA_VISIBLE_DEVICES"), "login-side device selection is prohibited; Slurm must assign it")


def single_gpu_token(value):
    require(isinstance(value, str) and value == value.strip() and value,
            "missing/ambiguous scheduler GPU token")
    require(re.fullmatch(r"[0-9]+", value) is not None or re.fullmatch(GPU_UUID, value) is not None,
            "GPU token must be one explicit decimal global ID or complete GPU UUID; no ranges/lists/MIG guessing")
    return str(int(value)) if value.isdigit() else value.lower()


def allocated_binding(environment):
    job = environment.get("SLURM_JOB_ID", "")
    step = environment.get("SLURM_STEP_ID", "")
    require(re.fullmatch(r"[1-9][0-9]*", job) is not None and re.fullmatch(r"[0-9]+", step) is not None,
            "requires this fresh numeric Slurm job and ordinary numeric step")
    for alias, value in (("SLURM_JOBID", job), ("SLURM_STEPID", step)):
        require(not environment.get(alias) or environment[alias] == value, "Slurm ID aliases disagree")
    tokens = [single_gpu_token(environment[key]) for key in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS") if environment.get(key)]
    require(tokens and len(set(tokens)) == 1, "job/step GPU assignment is absent, multiple or contradictory")
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    single_gpu_token(visible)
    require(environment.get("SLURM_CPUS_PER_TASK") == "4" and environment.get("SLURM_NTASKS") == "1",
            "task CPU/count differs from the frozen allocation")
    return {"job_id": job, "step_id": step, "global_gpu_id": tokens[0], "inherited_cuda_visible_devices": visible}


def parse_scontrol(text):
    matches = re.findall(r"(?:^|\s)([A-Za-z_][A-Za-z0-9_/:]*)=(.*?)(?=\s[A-Za-z_][A-Za-z0-9_/:]*=|$)", text.strip())
    result = {}
    for key, value in matches:
        require(key not in result, "duplicate scontrol field: " + key)
        result[key] = value.strip()
    require(result, "empty/unparseable scontrol reply")
    return result


def tres(text):
    result = {}
    for token in text.split(","):
        fields = token.split("=", 1)
        require(len(fields) == 2 and fields[0] not in result, "malformed/duplicate allocated TRES")
        result[fields[0]] = fields[1]
    return result


def memory_mib(value):
    match = re.fullmatch(r"([0-9]+)([KMGT]?)", value)
    require(match is not None, "unknown allocated memory unit")
    return int(match.group(1)) * {"": 1, "K": 1 / 1024., "M": 1, "G": 1024, "T": 1024 ** 2}[match.group(2)]


def time_seconds(value):
    parts = value.split(":")
    require(len(parts) in (2, 3) and all(part.isdigit() for part in parts), "unknown/unbounded Slurm time limit")
    numbers = [int(part) for part in parts]
    if len(numbers) == 2:
        numbers.insert(0, 0)
    require(numbers[1] < 60 and numbers[2] < 60, "malformed Slurm time limit")
    return numbers[0] * 3600 + numbers[1] * 60 + numbers[2]


def verify_job(job, step, binding, job_name, uid):
    require(job.get("JobId") == binding["job_id"] and step.get("StepId") == binding["job_id"] + "." + binding["step_id"],
            "scontrol returned a different job/step")
    require(job.get("JobName") == job_name, "allocation is not this launcher's fresh named job")
    owner = re.fullmatch(r"[^()]+\(([0-9]+)\)", job.get("UserId", ""))
    require(owner is not None and int(owner.group(1)) == uid, "allocated job is not owned by the current user")
    require(job.get("Account") == "xulab" and job.get("Partition") == "boron" and job.get("QOS") == "normal",
            "scheduler account/partition/QOS differs from the reviewed request")
    require(job.get("JobState") == "RUNNING" and step.get("State") == "RUNNING", "job/step is not running")
    require(job.get("NumNodes") == "1" and job.get("NumCPUs") == "4" and job.get("NumTasks") == "1",
            "actual allocation is not one node/task/four CPUs")
    require(time_seconds(job.get("TimeLimit", "")) == 900, "allocation time limit must be exactly 900 seconds")
    allocation = tres(job.get("AllocTRES", ""))
    require(allocation.get("cpu") == "4" and allocation.get("node") == "1" and memory_mib(allocation.get("mem", "")) == 32768,
            "actual allocation CPU/node/memory differs from 4/1/32 GiB")
    gpu = {key: value for key, value in allocation.items() if key == "gres/gpu" or key.startswith("gres/gpu:")}
    require(gpu and all(value == "1" for value in gpu.values()) and set(gpu) <= {"gres/gpu", "gres/gpu:g4090"},
            "actual allocation is not exactly one g4090 GPU")
    typed = job.get("TresPerNode", job.get("TRESPerNode", ""))
    require("gres/gpu:g4090" in gpu or typed == "gres/gpu:g4090:1", "missing explicit g4090 type evidence")


def gpu_query_argv(global_id):
    single_gpu_token(global_id)
    return [NVIDIA_SMI, "-i", global_id, "--query-gpu=index,uuid,name,driver_version,pci.bus_id,memory.total,memory.used",
            "--format=csv,noheader,nounits"]


def parse_gpu_reply(text, binding):
    rows = list(csv.reader(text.strip().splitlines()))
    require(len(rows) == 1 and len(rows[0]) == 7, "allocated-ID GPU query did not return exactly one complete GPU")
    index, device_uuid, name, driver, pci, total, used = [item.strip() for item in rows[0]]
    require(index.isdigit() and re.fullmatch(GPU_UUID, device_uuid) is not None,
            "invalid GPU index/UUID returned by the allocated-ID query")
    assigned = binding["global_gpu_id"]
    require((str(int(index)) == assigned) if assigned.isdigit() else device_uuid.lower() == assigned,
            "GPU query identity differs from scheduler assignment")
    visible = binding["inherited_cuda_visible_devices"]
    if visible.lower().startswith("gpu-"):
        require(visible.lower() == device_uuid.lower(), "inherited CUDA UUID differs from the assigned GPU")
    else:
        # Zero is accepted only when actually inherited from Slurm (cgroup
        # ordinal remapping), never fabricated or selected by this launcher.
        require(int(visible) in (0, int(index)), "inherited CUDA ordinal is not an unambiguous one-device binding")
    require(total.isdigit() and used.isdigit() and int(total) > 0 and name and driver and pci, "incomplete GPU/driver record")
    return {"index": int(index), "uuid": device_uuid, "name": name, "driver_version": driver,
            "pci_bus_id": pci, "total_memory_mib": int(total), "used_memory_mib": int(used)}


def clean_host_environment(environment, private, container=True):
    result = {key: value for key, value in environment.items()
              if not key.startswith(("APPTAINER_", "APPTAINERENV_", "SINGULARITY_", "SINGULARITYENV_", "SRUN_", "SBATCH_"))}
    result.update(APPTAINER_TMPDIR=private["apptainer_tmp"], APPTAINER_CACHEDIR=private["apptainer_cache"],
                  PYTHONDONTWRITEBYTECODE="1", TZ="UTC")
    if container:
        result.update(PATH=str(APPTAINER_PREFIX / "bin") + os.pathsep + result.get("PATH", "/usr/bin:/bin"),
                      LD_LIBRARY_PATH=str(APPTAINER_PREFIX / "lib"))
    return result


def container_argv(plan, gpu=None, allocation_env=None):
    private = plan["private"]
    env = {"HOME": private["home"], "TMPDIR": "/tmp", "XDG_CACHE_HOME": private["cache"],
           "TORCH_HOME": private["cache"] + "/torch", "CUDA_CACHE_PATH": private["cache"] + "/cuda",
           "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "GH_CUDA_PREINITIALIZE": "0", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
           "PYTHONPATH": str(ROOT) + os.pathsep + str(TEST_ENV / "lib/python3.11/site-packages"),
           "JAX_PLATFORMS": "cpu", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TZ": "UTC",
           "CUDA_VISIBLE_DEVICES": "", "NVIDIA_VISIBLE_DEVICES": "void"}
    argv = [str(APPTAINER), "exec", "--cleanenv", "--no-eval", "--no-home", "--no-mount", "hostfs,bind-paths",
            "--pwd", str(ROOT)]
    if gpu is not None:
        binding = allocated_binding(allocation_env)
        env.update({key: allocation_env[key] for key in SLURM_FIELDS if allocation_env.get(key)})
        # The independently verified scheduler-assigned physical UUID is the
        # only CUDA target. Preserve inherited ordinals in allocation.json;
        # never rely on cgroup remapping before the first CUDA context exists.
        env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
        env["NVIDIA_VISIBLE_DEVICES"] = gpu["uuid"]
        argv.append("--nv")
    binds = [(str(DELIVERY), str(DELIVERY), "rw"), (str(ROOT), str(ROOT), "ro"),
             (str(TEST_ENV), str(TEST_ENV), "ro"), (private["tmp"], "/tmp", "rw"),
             (private["var_tmp"], "/var/tmp", "rw")]
    for filename in sorted(set(plan["readonly_files"])):
        binds.append((filename, filename, "ro"))
    binds.append((plan["parsed_input"], plan["parsed_input"], "ro"))
    for source, destination, mode in binds:
        safe_path(source)
        argv.extend(["--bind", source + ":" + destination + ":" + mode])
    for key, value in sorted(env.items()):
        argv.extend(["--env", key + "=" + value])
    argv.extend([str(SIF), CONTAINER_PYTHON])
    return argv, env


def submission_argv(plan, launch_file, launch_sha):
    require(re.fullmatch(r"[0-9a-f]{64}", launch_sha) is not None, "invalid launch manifest hash")
    return [SRUN, "--account=xulab", "--partition=boron", "--qos=normal", "--nodes=1", "--ntasks=1",
            "--gres=gpu:g4090:1", "--cpus-per-task=4", "--mem=32G", "--time=00:15:00", "--immediate=60",
            "--quit-on-interrupt", "--kill-on-bad-exit=1", "--job-name=" + plan["job_name"],
            "--chdir=" + str(ROOT), HOST_PYTHON, str(Path(__file__).resolve()), "_inside",
            "--launch", str(launch_file), "--launch-sha256", launch_sha]


def command_record(argv, directory, name, timeout=10, environment=None):
    started = time.monotonic()
    record = {"argv": argv, "started_utc": utc_now(), "timeout_seconds": timeout}
    try:
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                                timeout=timeout, env=environment, cwd=str(ROOT))
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired as exc:
        def decoded(value):
            return value.decode("utf-8", "replace") if isinstance(value, bytes) else (value or "")
        record.update(returncode=None, stdout=decoded(exc.stdout), stderr=decoded(exc.stderr), error="timeout")
    except OSError as exc:
        record.update(returncode=None, stdout="", stderr=str(exc), error=type(exc).__name__)
    record["wall_seconds"] = time.monotonic() - started
    write_json(Path(directory) / (name + ".json"), record)
    return record


def verify_frozen(path, expected_hash):
    path = safe_path(path)
    require(digest(path) == expected_hash, "frozen workload changed")
    frozen = json.loads(path.read_text())
    require(frozen.get("schema") == 1 and safe_path(frozen["root"]) == ROOT, "frozen candidate root/schema differs")
    require(beneath(ROOT, DELIVERY) and beneath(path, DELIVERY) and beneath(frozen["output"], DELIVERY),
            "source and evidence must be in the new cloud-20261104 delivery")
    require(not beneath(frozen["output"], ROOT), "evidence must be outside the readonly source checkout")
    budget = frozen["protocol"]["budget"]
    require(budget["gpus"] == 1 and budget["cpus"] == 4 and budget["memory_gib"] == 32
            and budget["allocation_seconds"] == 900 and budget["maximum_gpu_hours"] == .25
            and budget["experiment_seconds"] == 840, "workload allocation budget changed")
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], universal_newlines=True).strip()
    require(head == frozen["head"], "candidate HEAD changed after CPU freeze")
    for relative, expected in frozen["source_sha256"].items():
        source = safe_path(ROOT / relative)
        require(beneath(source, ROOT) and digest(source) == expected, "frozen source changed: " + relative)
    for key, filename in frozen["assets"].items():
        expected = frozen["protocol"]["assets"][key]
        require(Path(filename).stat().st_size == expected["bytes"] and digest(filename) == expected["sha256"],
                "real pretrained/input asset differs from native CPU freeze")
    require(not (Path(frozen["output"]) / "execution.json").exists(), "frozen GPU experiment already attempted")
    return frozen


CPU_CHECK = """import json,os,sys,numpy,torch
from astrolabe.evaluators.torch_rng import capture_settings
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
print('CPU_CONTAINER_CHECK=' + json.dumps({'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),'numpy':numpy.__version__,'cuda_build':torch.version.cuda,'settings':capture_settings(),'cuda_initialized':torch.cuda.is_initialized(),'gpu_forward_calls':0,'environment':{k:os.environ.get(k) for k in ('HOME','TMPDIR','PYTHONPATH','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','GH_CUDA_PREINITIALIZE')}}))
"""


def prepare(args):
    reject_existing_allocation(os.environ)
    os.umask(0o077)
    frozen = verify_frozen(args.frozen, args.frozen_sha256)
    directory = safe_path(args.launch_dir)
    require(beneath(directory, DELIVERY) and not beneath(directory, ROOT) and not directory.exists(),
            "launch directory must be new and inside the task delivery, outside source")
    directory.mkdir(parents=True, mode=0o700)
    private = {}
    for key in ("tmp", "var_tmp", "home", "cache", "apptainer_tmp", "apptainer_cache"):
        path = directory / key
        path.mkdir(mode=0o700)
        private[key] = str(path)
    plan = {"schema": 1, "status": "PREPARING", "created_utc": utc_now(), "root": str(ROOT),
            "head": frozen["head"], "frozen": str(args.frozen.resolve()), "frozen_sha256": args.frozen_sha256,
            "directory": str(directory), "job_name": "astrolabe-gpu-20261104-" + uuid.uuid4().hex[:12],
            "private": private, "parsed_input": str(Path(frozen["output"]) / "parsed_input"),
            "readonly_files": [str(args.frozen.resolve()), frozen["assets"]["checkpoint"], frozen["assets"]["pdb"],
                               str(safe_path(frozen["protocol"]["assets"]["pdb"]["path"]))],
            "host_python": HOST_PYTHON, "container_python": CONTAINER_PYTHON,
            "source_sha256": digest(__file__), "requested_gpu_hours_ceiling": .25,
            "scientific_scope": "The reviewed fixed 30-decode SAME-candidate mutator control only; no additional warmup."}
    write_json(directory / "launch.json", plan)
    try:
        for executable in (HOST_PYTHON, SRUN, SCONTROL, SACCT, str(APPTAINER)):
            require(Path(executable).is_file() and os.access(executable, os.X_OK), "required site executable missing: " + executable)
        require(TEST_ENV.is_dir(), "CPU-preflight test-env directory is missing")
        require(SIF.stat().st_size == SIF_BYTES and digest(SIF) == SIF_SHA256, "actual SIF differs from the audited CPU environment")
        plan["sif"] = {"path": str(SIF), "sha256": SIF_SHA256, "stat": stat_identity(SIF)}
        plan["readonly_file_identity"] = {name: {"sha256": digest(name), "stat": stat_identity(name)} for name in plan["readonly_files"]}
        argv, environment = container_argv(plan)
        check = command_record(argv + ["-c", CPU_CHECK], directory, "cpu-container-check", timeout=60,
                               environment=clean_host_environment(os.environ, private))
        require(check["returncode"] == 0, "CPU-only container/mount validation failed; no GPU request permitted")
        lines = [line[len("CPU_CONTAINER_CHECK="):] for line in check["stdout"].splitlines() if line.startswith("CPU_CONTAINER_CHECK=")]
        require(len(lines) == 1, "ambiguous CPU-container validation record")
        actual = json.loads(lines[0])
        for key in ("python", "executable", "torch", "numpy", "cuda_build", "settings"):
            require(actual[key] == frozen["cpu_preflight"][key], "launcher CPU environment differs from native workload freeze: " + key)
        plan.update(status="READY_FOR_ROOT_REVIEW", cpu_container_check=actual, container_environment_cpu=environment)
    except BaseException as exc:
        plan.update(status="BLOCKED", error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        write_json(directory / "launch.json", plan)
    sha = digest(directory / "launch.json")
    print(json.dumps({"status": plan["status"], "launch": str(directory / "launch.json"), "sha256": sha,
                      "reviewable_srun_argv": submission_argv(plan, directory / "launch.json", sha),
                      "gpu_queries": 0, "slurm_jobs_submitted": 0}))


def read_launch(path, expected_hash):
    path = safe_path(path)
    require(beneath(path, DELIVERY) and digest(path) == expected_hash, "launch manifest changed/outside new task delivery")
    plan = json.loads(path.read_text())
    require(plan.get("status") == "READY_FOR_ROOT_REVIEW" and safe_path(plan["root"]) == ROOT,
            "launch has not passed CPU preparation")
    require(plan["source_sha256"] == digest(__file__), "launcher code changed after review")
    require(stat_identity(SIF) == plan["sif"]["stat"] and plan["sif"]["sha256"] == SIF_SHA256,
            "SIF changed since its pre-allocation full checksum")
    require(Path(plan["directory"]).resolve() == path.parent, "launch evidence directory mismatch")
    for name, expected in plan["readonly_file_identity"].items():
        require(stat_identity(name) == expected["stat"] and digest(name) == expected["sha256"], "readonly asset changed since preparation")
    return plan


def terminate_child(process):
    """Only this Popen child; Slurm owns containment of the allocation tree."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def allocation_clock(environment, job):
    clocks = []
    if environment.get("SLURM_JOB_START_TIME") or environment.get("SLURM_JOB_END_TIME"):
        start, end = environment.get("SLURM_JOB_START_TIME", ""), environment.get("SLURM_JOB_END_TIME", "")
        require(start.isdigit() and end.isdigit(), "incomplete scheduler epoch clock")
        clocks.append((int(start), int(end)))
    if job.get("StartTime") not in (None, "Unknown") and job.get("EndTime") not in (None, "Unknown"):
        require(environment.get("TZ") == "UTC", "scontrol date fallback requires the launcher's explicit UTC environment")
        clocks.append(tuple(int(datetime.datetime.strptime(job[key], "%Y-%m-%dT%H:%M:%S")
                                .replace(tzinfo=datetime.timezone.utc).timestamp()) for key in ("StartTime", "EndTime")))
    require(clocks and len(set(clocks)) == 1 and clocks[0][1] - clocks[0][0] == 900,
            "missing/contradictory exact 900-second scheduler allocation clock")
    return clocks[0]


def inside(args):
    os.umask(0o077)
    plan = read_launch(args.launch, args.launch_sha256)
    frozen = verify_frozen(Path(plan["frozen"]), plan["frozen_sha256"])
    directory = Path(plan["directory"])
    require(not (directory / "allocation.json").exists(), "allocation already entered; replay prohibited")
    record = {"status": "STARTING", "entered_utc": utc_now(), "entered_unix": time.time(),
              "host": os.uname().nodename, "host_python": sys.version,
              "environment": {key: os.environ.get(key) for key in SLURM_FIELDS + ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")},
              "launch_sha256": args.launch_sha256, "container_started": False}
    write_json(directory / "allocation.json", record)
    process = None
    binding = None
    result_code = 1
    try:
        binding = allocated_binding(os.environ)
        require(os.environ.get("SLURM_JOB_NAME") == plan["job_name"], "not the unique newly requested job")
        record["binding"] = binding
        write_json(directory / "allocation.json", record)
        job = command_record([SCONTROL, "--oneliner", "show", "job", binding["job_id"]], directory, "scontrol-job-before")
        step = command_record([SCONTROL, "--oneliner", "show", "step", binding["job_id"] + "." + binding["step_id"]], directory, "scontrol-step-before")
        require(job["returncode"] == 0 and step["returncode"] == 0, "cannot establish own scheduler job/step identity")
        parsed_job, parsed_step = parse_scontrol(job["stdout"]), parse_scontrol(step["stdout"])
        verify_job(parsed_job, parsed_step, binding, plan["job_name"], os.getuid())
        record["scheduler_job"] = parsed_job
        record["scheduler_step"] = parsed_step
        start_epoch, end_epoch = allocation_clock(os.environ, parsed_job)
        record["allocation_start_unix"] = start_epoch
        record["scheduled_end_unix"] = end_epoch
        query = command_record(gpu_query_argv(binding["global_gpu_id"]), directory, "allocated-gpu-before")
        require(query["returncode"] == 0, "assigned GPU metadata query failed")
        gpu = parse_gpu_reply(query["stdout"], binding)
        record["gpu"] = gpu
        record["effective_cuda_visible_devices"] = gpu["uuid"]
        remaining = end_epoch - time.time()
        container_timeout = min(870., remaining - 30.)
        require(container_timeout >= frozen["protocol"]["budget"]["experiment_seconds"] + 10.,
                "insufficient allocation time for unchanged workload plus cleanup; no shortened fallback")
        argv, container_environment = container_argv(plan, gpu, os.environ)
        argv += [str(ROOT / "benchmarks/runtime/gpu_mutator_control.py"), "run", "--frozen", plan["frozen"],
                 "--frozen-sha256", plan["frozen_sha256"], "--allocated-gpu-uuid", gpu["uuid"]]
        record.update(container_argv=argv, container_environment=container_environment,
                      container_timeout_seconds=container_timeout, status="RUNNING", container_started=True)
        write_json(directory / "allocation.json", record)
        with (directory / "container.stdout.txt").open("w") as stdout, (directory / "container.stderr.txt").open("w") as stderr:
            process = subprocess.Popen(argv, cwd=str(ROOT), env=clean_host_environment(os.environ, plan["private"]),
                                       stdout=stdout, stderr=stderr)
            result_code = process.wait(timeout=container_timeout)
        record.update(container_exit_code=result_code, status="CONTAINER_FINISHED" if result_code == 0 else "FAILED")
    except BaseException as exc:
        record.update(status="FAILED", error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        terminate_child(process)
        if process is not None:
            record["container_exit_code"] = process.poll()
        result_code = 1
    finally:
        record["step_wrapper_finished_utc"] = utc_now()
        record["step_wrapper_finished_unix"] = time.time()
        record["terminal_allocation_proof"] = "Not yet available inside a live step; require outer sacct after srun exits."
        write_json(directory / "allocation.json", record)
        if binding is not None:
            command_record([SCONTROL, "--oneliner", "show", "job", binding["job_id"]], directory, "scontrol-job-after", timeout=5)
            command_record([SCONTROL, "--oneliner", "show", "step", binding["job_id"] + "." + binding["step_id"]], directory, "scontrol-step-after", timeout=5)
    return result_code


def accounting_summary(text, job_id):
    rows = [row for row in csv.reader(text.splitlines(), delimiter="|") if row and row[0] == job_id]
    require(len(rows) == 1 and len(rows[0]) == 7, "terminal own-job accounting row is absent/ambiguous")
    job, state, code, start, end, elapsed, alloc = rows[0]
    require(state.split()[0] in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"},
            "job accounting is not terminal")
    require(start not in ("", "Unknown", "None") and end not in ("", "Unknown", "None") and elapsed.isdigit(),
            "actual allocation start/end/elapsed is unknown")
    allocation = tres(alloc)
    require(allocation.get("gres/gpu", allocation.get("gres/gpu:g4090")) == "1", "accounted GPU count is not one")
    return {"job_id": job, "state": state, "exit_code": code, "actual_start": start, "actual_end": end,
            "actual_elapsed_seconds": int(elapsed), "allocated_tres": allocation,
            "actual_gpu_hours": int(elapsed) / 3600., "within_gpu_budget": int(elapsed) <= 900}


def submit(args):
    reject_existing_allocation(os.environ)
    os.umask(0o077)
    plan = read_launch(args.launch, args.launch_sha256)
    verify_frozen(Path(plan["frozen"]), plan["frozen_sha256"])
    directory = Path(plan["directory"])
    path = directory / "submission.json"
    with path.open("x") as handle:
        handle.write("{\"status\": \"CLAIMED_ONCE\"}\n")
    argv = submission_argv(plan, args.launch.resolve(), args.launch_sha256)
    record = {"status": "SUBMITTING", "submitted_utc": utc_now(), "submitted_unix": time.time(),
              "srun_argv": argv, "launch_sha256": args.launch_sha256, "existing_job_reuse": False}
    write_json(path, record)
    process = None
    result_code = 1
    try:
        with (directory / "srun.stdout.txt").open("w") as stdout, (directory / "srun.stderr.txt").open("w") as stderr:
            process = subprocess.Popen(argv, cwd=str(ROOT), env=clean_host_environment(os.environ, plan["private"], container=False),
                                       stdout=stdout, stderr=stderr)
            result_code = process.wait(timeout=1020)  # 60 s admission + 900 s job + bounded client cleanup.
        record.update(srun_exit_code=result_code, status="SRUN_FINISHED")
    except BaseException as exc:
        record.update(status="INTERRUPTED_OR_FAILED", error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        terminate_child(process)  # Only our newly created srun client; its signals target its own tasks.
        if process is not None:
            record["srun_exit_code"] = process.poll()
        result_code = 1
    finally:
        record["srun_finished_utc"] = utc_now()
        record["srun_finished_unix"] = time.time()
        record["client_wall_seconds"] = record["srun_finished_unix"] - record["submitted_unix"]
        allocation_path = directory / "allocation.json"
        if allocation_path.exists():
            allocation = json.loads(allocation_path.read_text())
            binding = allocation.get("binding", {})
            job_id = binding.get("job_id", "")
            if re.fullmatch(r"[1-9][0-9]*", job_id):
                command = [SACCT, "--noheader", "--parsable2", "--jobs=" + job_id,
                           "--format=JobIDRaw,State%30,ExitCode,Start,End,ElapsedRaw,AllocTRES%200"]
                record["own_job_id"] = job_id
                record["terminal_accounting_argv"] = command
                accounting = command_record(command, directory, "sacct-own-job-final", environment={**os.environ, "TZ": "UTC"})
                try:
                    require(accounting["returncode"] == 0, "sacct query failed")
                    record["terminal_accounting"] = accounting_summary(accounting["stdout"], job_id)
                except ValueError as exc:
                    record["terminal_accounting_unknown"] = str(exc)
        terminal = record.get("terminal_accounting", {})
        if not terminal or not terminal.get("within_gpu_budget"):
            record["status"] = "PARTIAL_ACCOUNTING_OR_BUDGET"
            result_code = 2 if result_code == 0 else result_code
        elif terminal.get("state") != "COMPLETED" or terminal.get("exit_code") != "0:0":
            record["status"] = "FAILED"
            result_code = 1
        else:
            record["status"] = "COMPLETE" if result_code == 0 else "FAILED"
        write_json(path, record)
    print(json.dumps({"status": record["status"], "own_job_id": record.get("own_job_id"), "submission": str(path)}))
    return result_code


def main():
    def interrupted(signum, _frame):
        raise RuntimeError("launcher interrupted by signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command")
    p = subs.add_parser("prepare", help="CPU-only check and full SIF hash before allocation; no Slurm submission")
    p.add_argument("--frozen", type=Path, required=True)
    p.add_argument("--frozen-sha256", required=True)
    p.add_argument("--launch-dir", type=Path, required=True)
    for command in ("submit", "_inside"):
        p = subs.add_parser(command)
        p.add_argument("--launch", type=Path, required=True)
        p.add_argument("--launch-sha256", required=True)
    args = parser.parse_args()
    require(args.command is not None, "choose prepare or submit explicitly")
    if args.command == "prepare":
        prepare(args)
        return 0
    if args.command == "submit":
        return submit(args)
    return inside(args)


if __name__ == "__main__":
    raise SystemExit(main())
