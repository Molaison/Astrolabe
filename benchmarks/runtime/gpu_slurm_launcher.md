# Bounded task-specific Slurm launcher

This launcher is separate from the already reviewed scientific harness. It does
not change the 4GYT input, model, precision, seeds, requests or scientific gates.
The host entrypoint is `/usr/bin/python3` (f102 Python 3.6.8); the container uses
the same `/alphafold3_venv/bin/python`, SIF and `test-env` as the native CPU
preflight. No new packages, weights or images are downloaded.
OMP, MKL, OpenBLAS and NumExpr threads all remain one, matching root's CPU
freeze wrapper. `GH_CUDA_PREINITIALIZE=0` prevents the old profile's eager CUDA
initialization; HIP/ROCR visibility is empty in both modes. CPU preparation has
no visible accelerator, while the GPU run exposes only the verified assigned UUID.

## Preconditions and exact entrypoints

Root first selects the final source candidate, commits it, and completes the
native CPU `gpu_mutator_control.py freeze`. Its frozen input and output directory
must be under the new `cloud-20261104`, outside the readonly source checkout.
Do not prepare the launcher against an intermediate candidate: changing source,
HEAD, inputs or launcher after preparation invalidates the manifest.

From the final remote `repo-offline` checkout, on the login node with no inherited
Slurm allocation or selected CUDA device:

```bash
python3 benchmarks/runtime/gpu_slurm_launcher.py prepare \
  --frozen /xcf26/zpzeng/practice_arena/astrolabe_runtime_20261102_implementation/cloud-20261104/gpu-control-01/frozen.json \
  --frozen-sha256 FROZEN_SHA256 \
  --launch-dir /xcf26/zpzeng/practice_arena/astrolabe_runtime_20261102_implementation/cloud-20261104/gpu-allocation-01
```

`prepare` hashes the real 6.4 GB existing SIF **before requesting any GPU**, checks
frozen source/assets, creates private directories with umask 077, and runs a
CPU-only container/mount/environment check. That check has `CUDA_VISIBLE_DEVICES`
empty, `NVIDIA_VISIBLE_DEVICES=void`, no `--nv`, zero model forwards and zero
Slurm requests. It must match the freeze's Python, Torch, NumPy, CUDA build and
numerical settings. Full stdout/stderr and failures are retained. It prints a
launch-manifest SHA and the exact future `srun` argv for root review.

Only after root's final gate, execute **once**:

```bash
python3 benchmarks/runtime/gpu_slurm_launcher.py submit \
  --launch /xcf26/zpzeng/practice_arena/astrolabe_runtime_20261102_implementation/cloud-20261104/gpu-allocation-01/launch.json \
  --launch-sha256 PRINTED_LAUNCH_SHA256
```

The resource request is fixed:

```text
srun --account=xulab --partition=boron --qos=normal --nodes=1 --ntasks=1
     --gres=gpu:g4090:1 --cpus-per-task=4 --mem=32G --time=00:15:00
     --immediate=60 --quit-on-interrupt --kill-on-bad-exit=1
     --job-name=<unique task nonce> --chdir=<final source>
     /usr/bin/python3 <this launcher> _inside --launch <manifest> --launch-sha256 <hash>
```

This is one new job, with no node pin, physical GPU selection, `--jobid`, node
exclusivity, queue fallback, retry, or operation on the user's existing jobs.
The scheduler must admit the request; stale inventory is not ownership proof.
Account/QOS permission, shared filesystem visibility, executable Apptainer,
working driver binds and enough remaining allocation time are real prerequisites.
An unavailable resource or failed prerequisite leaves evidence and stops.

## Allocation, binding and containment

Inside the newly named job, the launcher queries only `scontrol show job JOBID`
and `scontrol show step JOBID.STEPID`. It verifies the current UID, unique name,
account, QOS, partition, running state, one node/task, four CPUs, 32 GiB, one
g4090 and the exact 900-second time limit. Its actual allocation clock comes
from Slurm epoch fields and/or its own UTC `scontrol` timestamps; conflicting or
unknown clocks fail. Container startup, all failures and cleanup count against
this allocation. Thirty seconds are reserved for outer cleanup; insufficient
time for the unchanged experiment plus startup causes a recorded failure.

`SLURM_STEP_GPUS` or `SLURM_JOB_GPUS` must name exactly one global ID or complete
UUID. If both are supplied, they must agree. Missing IDs, ranges, lists, partial
UUIDs or contradictions fail before any GPU query. Only then is `nvidia-smi -i
ASSIGNED_ID` called, querying that single GPU's UUID, index, name, driver, PCI
identity and memory. There is no login-node GPU query and no all-device query.

The inherited single `CUDA_VISIBLE_DEVICES` value is retained as provenance.
An inherited zero can be Slurm's cgroup-local ordinal; the launcher does not
assume that remapping is enforced correctly. After independently resolving the
scheduler's global ID, both container `CUDA_VISIBLE_DEVICES` and
`NVIDIA_VISIBLE_DEVICES` are set to the actual assigned UUID, never zero or
`all`. This binds the intended GPU before the first CUDA context is created.
The same UUID is passed to the harness for its additional actual-property check.

Apptainer uses `--nv --cleanenv --no-eval --no-home`; automatic hostfs/site binds
are disabled, explicit source/test-env/checkpoint/input mounts are readonly,
and only the new delivery directory is writable. Container `/tmp` and `/var/tmp`
map to task-private directories; HOME, caches and Apptainer session/cache paths
also stay there. The launcher removes inherited Apptainer/Singularity bind/env
overrides. It creates no PID namespace, preserving the existing harness's
process-identity observer. Host Slurm client libraries are retained; only the
Apptainer subprocess receives its installed runtime-library prefix.

`--nv` alone is not device isolation, and NVIDIA's visibility variable alone is
not ownership proof in Apptainer. Device ownership depends on the verified Slurm
allocation and its enforcement; CUDA visibility and the harness UUID gate are
additional checks. No model/GPU operation is performed outside that allocation.

## Evidence and terminal release

The launch directory retains `launch.json`, the full CPU check, submission and
container argv/stdout/stderr, safe allocation environment fields, before/after
own-job/step `scontrol` records, the scoped GPU query, and `allocation.json`.
Existing harness output stays in its frozen directory, including independent
process release and the raw RSS/VRAM/GPU timeline. Private exchange directories
are preserved after uncertain outcomes; the launcher does not delete evidence.

Inside-step completion is not terminal allocation release. After its own `srun`
exits, the outer launcher queries `sacct` **only for the recorded new job ID** and
retains the complete raw reply. The top-level job row supplies actual start,
end, exit/state, allocated TRES and elapsed seconds. Actual one-GPU cost is that
allocation elapsed time divided by 3600; client queue time and the harness's
inner timer are reported separately. Elapsed allocation time over 900 seconds
is a budget failure even if the scheduler allowed shutdown grace/overrun.

Delayed or unavailable accounting yields `PARTIAL_ACCOUNTING_OR_BUDGET`, never
an invented release. The exact own-job `sacct` argv remains in `submission.json`
for root's later readonly terminal verification. A separate successful terminal
record can close that evidence gap without rerunning scientific work. Failed or
incomplete GPU experiments must remain in the final report.

## Validation scope and primary references

Local tests use synthetic scheduler records and forbid subprocess launch. They
cover missing/foreign/multiple device assignments, inherited ordinal remapping,
wrong resource budgets, safe argv/binds, CPU-only preflight, Python 3.6 syntax,
own-job accounting and overrun rejection. They do not prove scheduler admission,
GPU availability, pretrained inference, or actual resource release.

The installed f102 CLI was read without allocation: Slurm 26.05.4 and Apptainer
1.5.2 advertise the required flags. Raw successful help and initial transient
`runtime_unavailable` failure are retained in `control/gpu-launcher-review/`.

Primary semantics: [Slurm srun](https://slurm.schedmd.com/srun.html),
[Slurm prolog/epilog GPU binding](https://slurm.schedmd.com/prolog_epilog.html),
[Apptainer exec](https://apptainer.org/docs/user/main/cli/apptainer_exec.html),
[Apptainer GPU visibility](https://apptainer.org/docs/user/latest/gpu.html).
