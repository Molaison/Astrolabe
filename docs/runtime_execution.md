# Isolated trajectories and shared execution admission

The current implementation and operator instructions are documented in
[runtime_execution_guide.md](runtime_execution_guide.md). The preserved baseline
is recorded in [runtime_implementation_baseline.md](runtime_implementation_baseline.md);
exact evaluator and mutator variants and their evidence limits are in
[evaluator_lifecycle_matrix.md](evaluator_lifecycle_matrix.md) and
[mutator_lifecycle_matrix.md](mutator_lifecycle_matrix.md).

## Current contract

`execution.enabled: true` runs the already configured repeats in separate
GPU-hidden trajectory processes. The original repeat seeds and scientific state
writers remain unchanged. Admission starts at one independent trajectory and
uses bounded observation windows, hysteresis, resource pressure, failures and
backend/shape completion rates to adjust in-flight work. Complete trajectory
completion counts are recorded separately from short-window phase proxies.
The default remains `execution.enabled: false`.

Both evaluator and mutator workers use the same broker ledger. Allocated CUDA
tokens are resolved to physical UUIDs; a per-user OS lock prevents competing
Astrolabe owners in the same shared host lock domain. Containers must share that
domain or have an exclusive outer allocation. Other users and unrelated
programs remain the external scheduler's responsibility. Unknown allocation,
identity, telemetry or residency never authorizes additional GPU capacity.
Same-card device execution remains serial; no cross-request batching has been
numerically certified.

Worker registration limits (`max_resident_models`, `max_resident_per_device`)
are separate from resident/scratch VRAM, CPU/RAM, prepared requests and output
budgets. The CLI exposes these limits and `execution.resources` /
`execution.control`. Defaults and units are in the operator guide.
Managed handlers yield the device slot during audited CPU stages while retaining
their request's memory/output reservation. Composite native calls retain both
CPU and GPU admission. Related embedding models and known external subprocesses
remain in the admitted process group.

Process liveness, successful weight loads, tensor/device residency and readiness
are separate lifecycle fields. Exact model identity includes actual code,
checkpoint, interpreter/package/native library environment, relevant initialization
settings and precision. Unobserved fields stay unknown. A live daemon is not
a load or residency certificate.

## Failure and containment

Client/worker epochs, transport request IDs and scientific identities fence
duplicate, out-of-order and late responses. Outcome-unknown errors are preserved
end-to-end and never trigger an automatic scientific retry. The native owner
retains accepted revisions, stage transactions, fixed/shared segments and RNG.
Terminal completion is recorded only after requested release is confirmed.

A guardian holds the physical-device owner lock across broker death. CUDA-capable
worker bootstrap registers its PID/start time before model imports; known child
processes inherit the managed group. Group cleanup is anchored to process
identities, so stale numeric PIDs/PGIDs are not blindly killed. Unconfirmed
cleanup retains artifacts and admission ownership.

Arbitrary third-party double-fork/session-detaching behavior is outside this
process-group certification. Such a backend needs job-cgroup containment and
explicit runtime evidence before its entire descendant lifetime can be certified.
It must not be declared released merely because an opaque bridge exited.
Standard tree-codec byte limits are enforced before publication; vendor-written
files, RSS and VRAM use sampled breach detection/termination, not a filesystem
or kernel memory hard quota.

## Supported and rejected components

The actual factory/registry and adapter declarations define the backend range.
All 13 evaluator families and the 20 mutator registry names are inventoried in
the matrices, including CPU-only implementations and exact MPNN/Protenix variants.
A registered bridge or implementation contract is not GPU numerical certification.
New Boltz2 and unspecified native callback versions are not silently substituted.

The runtime checks active mutators/evaluators/gate evaluators, configured losses
and implicit loss expressions, selectors and hooks. Active structure starts,
predockers, GPU-capable per-residue plugins and unreviewed losses/selectors/hooks
still fail closed until their resource boundary is connected. Selector admission
pins the reviewed dispatch and implementation digests, not merely a class name.

`CPUExecutor` remains a separately usable bounded, GPU-hidden subprocess utility.
It is not the trajectory scoring/preparation call chain; those stages stay in
their existing trajectory owner. The implementation does not claim to have moved
every CPU operation to a new pool.

## Evidence and comparison

`scripts/validation/runtime_workflow_benchmark.py` interleaves fixed direct,
resident-serial and adaptive workloads, with an explicit overall time/output
budget, external timing, score/decision and emitted-coordinate comparisons.
`--direct-repo` selects a frozen old checkout; otherwise the baseline is explicitly
the current checkout's legacy path. Raw logs, traces, source manifests and
integration results live in `evidence/runtime-20261103/`.

GPU activity is diagnostic. Effective completed scientific work and resource
cost determine the result. CPU fixtures, synthetic checkpoints, import tests and
mock-device tests do not certify trained GPU models, cache equivalence or speed.
Real GPU certification remains configuration-specific and requires an allocated
device, exact installed versions/weights and an explicit experiment budget.
