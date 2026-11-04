# ProteinMPNN GPU control — protocol and execution

This is a same-candidate comparison of Astrolabe's existing direct native wrapper
(`legacy`) and existing private process runner (`resident`). Both own one loaded
model for three sequential requests. It certifies neither a pretrained evaluator
nor shared broker admission, MCMC acceptance, or a revision-level speedup. No
production model code, precision, input size or sampling algorithm changes here.

## Frozen workload

`workloads/proteinmpnn_gpu_control.json` specifies every `design_sequences`
argument, constructor options, assets and native source hashes. It uses the
actual `proteinmpnn_v_48_020.pt` pretrained checkpoint and the full 4GYT A/B input:
177 residues per chain, 354 total, backbone masks all one, native sequence SHA
`a149f24cd9a0ca21eaeed7478450b91dfec6800b5d372e06655511e6b49a8e44`.
The bundled model source is unchanged from baseline `d66ee714` to `9d62b3eb`.

Seeds 17–21 define five groups, alternating legacy/resident order. Each of ten
fresh trial processes constructs exactly one model and calls `design_sequences`
three times, without request reseeding or warmup. This is **ten model loads,
thirty native sample calls**, and five additional existing resident device
identity probes that do not run the model. The native constructor itself seeds
its RNGs; the harness performs the same one-time initial seeding in both modes.

Temperature is 0.3, omitted amino acids are `[C]`, batch size and number of
batches are one, `parse_atoms_with_zero_occupancy=1`, parser chains are the real
list `[A, B]`, and both chains are designed. No symmetry, fixed residues,
per-residue biases, multistate data or teacher forcing are enabled. The native
ProteinMPNN constructor forces its unused side-chain context off; the config's
original context option is still recorded explicitly. Unused backend checkpoint
paths are empty and are never read.

## CPU preflight before reserving a GPU

`freeze` requires `CUDA_VISIBLE_DEVICES=''`. It verifies full checkpoint/input
hashes, actual native source, clean tracked candidate, complete native signature,
true CPU parsing, exact chain order/native sequence/all-one masks, and the real
native constructor loading 1,660,485 float32 parameters. It performs **no forward
or sample**. This is CPU readiness evidence, not GPU certification.

Use the already available SIF CPU entrypoint with readonly binds for the two
referenced asset directories. Work exclusively in the new delivery tree:

```bash
TASK_BASE=/xcf26/zpzeng/practice_arena/astrolabe_runtime_20261102_implementation
TASK_TREE="$TASK_BASE/cloud-20261104/repo-offline"
TASK_OUTPUT="$TASK_BASE/cloud-20261104/gpu-control-01"
APPTAINER_BINDPATH=/xcf26/zpzeng/gitrepo/ligandmpnn/model_params:/xcf26/zpzeng/gitrepo/ligandmpnn/model_params:ro,/xcf26/zpzeng/gitrepo/ligandmpnn/inputs:/xcf26/zpzeng/gitrepo/ligandmpnn/inputs:ro \
  "$TASK_BASE/test-container" "$TASK_TREE" \
  benchmarks/runtime/gpu_mutator_control.py freeze --output "$TASK_OUTPUT"
```

The final JSON contains the exact absolute paths, candidate HEAD/tree, source
hashes, CPU framework/interpreter versions, all planned trials and actual parsed
input. Save the printed SHA-256 of `frozen.json`. The output directory must be
new. Any failure retains its partial files and is not silently retried. Changes
to paths or a later candidate require an explicitly new freeze before allocation.
This example is an entrypoint, not a statement that it has already run.

## GPU execution after independent review and root allocation

Run once inside the separately prepared scheduler-bound GPU container, with the
same Python/SIF, assets, source and frozen output directory. The root workflow
allocates at most **one GPU, four CPUs, 32 GiB, 15 minutes / 0.25 GPU-hours**;
request creation has an immediate wait limit of 60 seconds. Do not reuse any
existing job. The harness requires a Slurm job ID and exactly one inherited CUDA
device, rejects `NVIDIA_VISIBLE_DEVICES=all` and shared broker credentials, and
independently requires actual CUDA availability plus a float32 CUDA-resident
model with load count one. It never selects another device or falls back to CPU.

```bash
python benchmarks/runtime/gpu_mutator_control.py run \
  --frozen "$TASK_OUTPUT/frozen.json" --frozen-sha256 EXACT_PRINTED_SHA256 \
  --allocated-gpu-uuid ROOT_VERIFIED_ALLOCATED_GPU_UUID
```

The experiment has a frozen 840-second ceiling (leaving allocation cleanup
headroom) and a 150-second maximum per trial. A failed process, incomplete
request, invalid scientific pair, forced cleanup or uncertain release stops
later trials. Completed and failed attempts retain their logs, raw data and wall
time. There is no resume, replay, output filtering, extra warmup, or shortened
fallback workload. If the budget cannot complete the fixed plan, certification
is `INVALID`, with partial measurements retained.

The harness reuses `runtime_workflow_benchmark.run_once`: independent process
completion remains the timing authority while RSS/CPU and UUID-scoped GPU
utilization/VRAM are sampled every 0.5 seconds into each trial's raw timeline.
The caller must supply the actual GPU UUID verified from the fresh allocation;
the trial compares it with the real CUDA device properties. Missing readings
remain unknown and yield `resource_telemetry_status=PARTIAL_UNKNOWN`, never zero.
The frozen output budget is 260,000,000 bytes across the evidence directory;
each trial receives only the remaining allowance. As in the reused harness,
this is an observed output limit, not an atomic filesystem quota. Partial files
and any observed overrun remain as failure evidence. Native private temporary
exchange and the full job's storage/resource ceiling remain the allocation
owner's responsibility. The root Slurm start/end record is required to account
for GPU allocation startup/shutdown outside the inner experiment timer.

The existing private runner does not contact a shared execution broker. Its
public methods perform prepare/execute/finalize. The direct path calls the same
native wrapper and materializes results with the existing `cpu_tree` helper.
Only an observational reference to the private client's process/lifecycle is
retained to verify public `close()`. No request bypasses the public runner API.

## Scientific and lifecycle gates

The actual native return consists of `S`, `sampling_probs`, `log_probs`, `logits`,
`decoding_order`, `chain_letters`, `alphabet` and `distribution_semantics`.
The unused native local `out_dict` contains other fields but is not returned.
The harness therefore obtains masks and the native sequence from an independent
CPU parse of the byte-identical frozen input. It does not claim those fields
were returned by GPU inference.

Every returned tensor must be CPU-ready, finite, have the full expected shape
and original dtype. The 354-element decode order must be a permutation. Amino
acid indices and chain/alphabet order must be valid; probabilities must be
nonnegative, normalized with absolute tolerance 1e-5, and assign zero to omitted
C. Every sampled sequence must omit C. The derived mean NLL is explicitly a CPU
recalculation of the native `get_score` formula using returned log probabilities
and the all-one input mask; it is not described as a returned native score.

All discrete outputs, Python/NumPy/Torch CPU/CUDA RNG states and numerical
settings must match exactly between paired modes after construction and after
each request. Before/after raw RNG snapshots also prove continuity between
requests and actual CUDA RNG advancement. Floating tensors use predeclared
`rtol=1e-5`, `atol=1e-6`; bitwise equality and maximum absolute/relative differences
are recorded even when tolerance checks pass. A failed gate is not relaxed.

The model must remain loaded exactly once and on `cuda:0` in float32 at all
request boundaries. Resident close must report confirmed worker release and an
actual child exit. Direct close must clear its model; final CUDA-context release
requires process exit. The existing `BenchmarkProcessObserver` independently
checks the original PID/start-time identities and observed descendants through
exit. Its observational limitation is retained: complete containment of an
arbitrary unobserved double-fork or namespace escape requires the allocation's
job cgroup. Root/A collect the separate scheduler allocation/release evidence.

## Measurements and interpretation

The primary rate is completed valid design requests divided by the sum of
spawn-to-observed-release wall times. It includes imports, hash checks, the
resident identity probe, model construction, serial calls, CPU output exchange,
evidence serialization and close. The full controller/allocation elapsed time
is also retained. This rate is not an isolated kernel metric or MCMC throughput.

Separate fields report public runner construction (including native model load
and resident startup/identity work), first request, and request positions 1/2 as
steady calls. Call timing includes actual CPU materialization and CUDA
synchronization, excluding subsequent evidence serialization; primary workflow
timing includes the latter. There are no hidden warmup requests. Five paired
workflow ratios and the per-mode raw totals are retained, including regressions.
Both modes already reuse models; resident is not presumed faster.

`comparison.json` may be `PASS` only when all ten trials, thirty requests, five
scientific pairs and release checks pass. A passing correctness certificate
does not itself imply a performance improvement. `INVALID` reports retain
diagnostic completed-per-second figures, but omit certified throughput metrics.
Any improvement applies only to this same-candidate mode contrast; attributing
an improvement to this turn's changes requires a distinct revision comparison.
Execution records must form the exact frozen ordered prefix, with all index,
ID, group, seed and mode fields intact; duplicates, extra records, reordering
and nonpositive/nonfinite trial time are rejected before aggregation.
Each child's own identity, candidate HEAD and frozen-manifest hash must also
match its planned controller entry. Request indices must be exactly 0/1/2 with
first/steady/steady phase labels. Phase times must be positive, finite and fit
within the observed workflow interval. A complete-looking stale child file or
duplicated request metadata therefore cannot certify effective throughput.
The original `measurement.process_release` and the convenient duplicated
`observation` must exist and agree in full, including the duplicated exit code.
Missing records, observation errors, unknown groups or contradictory release
flags invalidate the raw reread. A resource timeline must still exist to mark
resource telemetry complete; missing measurements or raw files remain unknown.

An independent CPU-only reread of the immutable raw evidence is available:

```bash
python benchmarks/runtime/gpu_mutator_control.py compare \
  --frozen "$TASK_OUTPUT/frozen.json" --frozen-sha256 EXACT_PRINTED_SHA256
```

Focused local tests use genuine CPU tensors with synthetic outputs/RNG metadata
to test fixed tolerances, malformed outputs, missing trials, negative scientific
controls, allocation guards and release failures. They are explicitly harness
contract tests, not pretrained inference or GPU evidence.
