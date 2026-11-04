# Exact ProteinMPNN runner control: GPU admission blocked

**BLOCKED_ADMISSION; GPU NOT_RUN.** The final source completed a genuine pretrained CPU constructor/input freeze and the same-SIF CPU launcher preparation. The single bounded Slurm request, job **33334**, waited for resources and then failed admission. No GPU container, trial, design request or comparison pair ran. Actual GPU allocation and GPU-hours were **zero**; there is no GPU correctness or performance result to certify.

This supplements the [pre-run backend matrix](runtime_backend_certification.md) and its [generated JSON snapshot](../evidence/runtime-next/backend-certification-final.json). The exact machine-readable result is [runner-certification-supplement.json](../evidence/runtime-next/gpu-control-01/runner-certification-supplement.json). Registry coverage remains **14 evaluator + 20 mutator keys; 28 BLOCKED / 6 PARTIAL / 0 DONE**. Readiness evidence does not promote a complete registry scope.

## What was completed

| Scope | Verdict | Actual evidence |
|---|---|---|
| Final implementation identity | **DONE** | Published HEAD `1bdaf3a53227c9197b930dc00a01b752acd90067`, tree `6f763bfb04778ef4f32386ddb8aeb38ce4240bf1`; B independently matched all 302 frozen source-file digests. |
| Real pretrained CPU constructor and full input | **DONE** | Native `LigandMPNN` constructor with `model_type=protein_mpnn`: one load, 1,660,485 float32 parameters on CPU; zero forward/sample calls, CUDA uninitialized. Full 4GYT A177+B177, finite `X[354,4,3]`, mask sum 354, exact native sequence and numerical settings. |
| CPU launcher preparation | **DONE, with recorded HOME limitation** | Fresh full checksum of the existing 6,418,354,176-byte SIF; identical Python/Torch/NumPy/settings; readonly source/assets and private temporary-directory bind arguments; no GPU request during preparation. |
| One bounded scheduler request | **DONE** | `srun` requested one g4090, four CPUs, 32 GiB, 900 seconds, admission wait at most 60 seconds, under xulab/boron/normal. No retry or existing-job reuse. |
| GPU admission and real end-to-end control | **BLOCKED** | Job 33334 was queued, then `Requested nodes are busy`; no inside-step allocation record or harness execution/trial files exist. |
| GPU output/RNG parity, resident reuse, five-pair throughput | **NOT_RUN** | Zero GPU trials, GPU model loads, design requests or comparison pairs. No throughput or speedup is inferred from CPU preparation or queue waiting. |
| Independent terminal accounting | **DONE** | Agent A read exact-job `sacct`/`scontrol`/`squeue`; B independently parsed those raw responses and matched the original launch nonce and UID. No actual start, node or allocated TRES; zero elapsed allocation time. |
| Original evidence preservation and CPU reread | **DONE** | All 27 original files, 977,895 bytes across 40 file/directory entries preserved losslessly. Every file digest and the archive digest verified; B reloaded the actual parsed-input tensors on CPU. |

See B's [source check](../evidence/runtime-next/final-head-1bdaf3a/agent-b/source-handoff-verification.json), [native freeze review](../evidence/runtime-next/final-head-1bdaf3a/agent-b/native-freeze-review.json), [launcher preparation review](../evidence/runtime-next/final-head-1bdaf3a/agent-b/native-launch-review.json), and [CPU artifact/accounting reread](../evidence/runtime-next/final-head-1bdaf3a/agent-b/cpu-and-admission-review.json). These reviews do not execute a model or GPU workload.

## Frozen scientific scope

The intended public API was `build_runner("ligandmpnn", model_type="protein_mpnn")`, followed by `design_sequences` and `close`. The comparison was the **same candidate's** existing legacy wrapper versus existing private resident runner; both would load one model and reuse it for three sequential requests. It was not a comparison against an earlier revision.

The [frozen protocol](../benchmarks/runtime/workloads/proteinmpnn_gpu_control.json), SHA-256 `72a18eef550fdf5555501b4163126522d752f441c3b7a5aad517a852d6498fc6`, retained the complete 354-residue input, seeds 17–21 in five alternating groups, temperature 0.3, omit C, batch 1×1, all native constructor/decode arguments, the original precision and fixed numerical tolerances. Ten model loads and thirty requests were **planned**, not performed. Requests were specified serially; no smaller input or changed setting was substituted, and no warmup or retry was added.

| Identity | SHA-256 |
|---|---|
| Actual `frozen.json` | `21460a82249fadffde75bc0ce058af40548e85daf7aa29851b2fea992e72310a` |
| Actual `launch.json` | `9bbbae547470afda98c361f6a6796e14c6ef0f71dab536b7a6f503ee2ba8a921` |
| ProteinMPNN v48_020 checkpoint | `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd` |
| Full 4GYT input PDB | `c05c06773ae8c00d2d033f7f7a103c4048981d79418c6c1a2579b15da455b2b7` |
| Parsed native sequence | `a149f24cd9a0ca21eaeed7478450b91dfec6800b5d372e06655511e6b49a8e44` |
| Same SIF, freshly hashed by actual preparation | `242aae3bb97c5c7ea374dab5d7462e04863d9d48ba2ee00cc57e6c0a57a538e3` |

The actual CPU environment was Python 3.11.14, `/alphafold3_venv/bin/python`, Torch 2.7.1+cu126 and NumPy 1.26.4. A CUDA-enabled build is not evidence of CUDA execution: the CPU freeze and preparation explicitly recorded CUDA uninitialized and no visible GPU. Native CPU freeze took 7.785164701 seconds; launcher preparation took 23.617934077 seconds. These are preparation costs, not inference throughput measurements.

## Admission failure and actual resource cost

The original stderr contains:

```text
srun: job 33334 queued and waiting for resources
srun: error: Unable to allocate resources: Requested nodes are busy
```

| Observation | Recorded result |
|---|---|
| Exact job identity | `33334`, `astrolabe-gpu-20261104-10cf2b34c104`, user `zpzeng`, UID `7054` |
| Submit / terminal UTC | 2026-11-04 05:51:06 / 05:52:06 |
| Terminal state | `CANCELLED by 7054` |
| Actual start / allocation elapsed | `sacct Start=None`, `ElapsedRaw=0` |
| Actual resources | `AllocCPUS=0`, empty `AllocTRES`, `NodeList=None assigned`; own-job queue empty |
| `srun` client exit / wall | Exit 1 / 60.107918978 seconds |
| Outer submission wrapper wall | 60.659714499 seconds |
| Actual GPU usage | **0 allocated GPUs; 0 GPU-hours** |

The requested 0.25 GPU-hour ceiling is not usage. The 60-second interval is queue waiting. `scontrol` retained a future predicted `StartTime`; it was excluded from actual duration/cost calculations. Its arithmetic node headroom and `Reason=Priority` do not establish that this particular request could start, or identify a unique scheduler cause. The observed admission failure is reported without claiming that every physical GPU was occupied.

The original launcher `submission.json` remains `PARTIAL_ACCOUNTING_OR_BUDGET`: `_inside` never ran, so the launcher had no recorded own-job ID for its automatic accounting step. [A's independent accounting proof](../evidence/runtime-next/final-head-1bdaf3a/agent-a/allocation-proof.md) and [raw-backed JSON](../evidence/runtime-next/final-head-1bdaf3a/agent-a/accounting-proof.json) close this uncertainty externally. No original record was edited to manufacture a successful launch. The accounting exit field `0:0` is also preserved; it does not supersede the CANCELLED state, `srun` exit 1 or absence of an allocation.

## Recorded environment limitation

Apptainer emitted `Overriding HOME environment variable with APPTAINERENV_HOME is not permitted`. Actual HOME remained `/xcf26/zpzeng`, rather than the requested private HOME. The CPU freeze also retained ProDy's warning that it could not write its configuration there. The report therefore claims only the recorded `--no-home`/readonly/private-temporary bind arguments, actual `TMPDIR=/tmp`, and explicit cache arguments; it does **not** claim that the actual HOME variable was private. No user-home write is evidenced. GPU container execution and its actual mounts were never reached.

## Raw data and independent reproduction

The [B evidence directory](../evidence/runtime-next/final-head-1bdaf3a/agent-b/README.md) contains the complete original archive, per-file manifest, verification records and a sealed archive of transport observations and the offline audit script.

- Raw archive: **126,328 bytes**, SHA-256 `5588acc27dceb7595fe7aff8be79385ec9186913aa7a486fc54e8ea0e750b8e9`.
- Original-file manifest: SHA-256 `2e620a82f9e7e08fc91fd212f50aa3a05585441fdeace9036d7f53a878caaa1b`.
- B independently downloaded every byte in bounded chunks, verified all 27 file hashes, matched the source/protocol, reloaded the CPU input tensors and parsed A's terminal-accounting captures. All original failures and warnings remain in the archive.

No GPU `aggregate` or pair-comparison result was produced: there are no GPU raw outputs or RNG snapshots to compare. Backend inventory and earlier CPU/synthetic contract evidence remain scoped as before. Full registry mutator selector/proposal/acceptance, evaluator W1, MCMC/stage transitions, shared broker behavior, other MPNN variants, real GPU parity and reproducible GPU throughput improvement remain unproved by this control.
