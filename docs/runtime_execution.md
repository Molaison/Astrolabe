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

Broker ingress uses separate bounded Unix-socket lanes for business work,
short control operations, and nested device admission. Business and nested
capacity follow `execution.resources.pending_items`; short control capacity
is eight concurrent connections. Nested admission cannot consume the short
control lane needed for cancellation, metrics, and detach. These bounds include
accepted sockets reading a frame. The accept loop never waits for a free
handler, and each frame has an absolute read deadline. Only one nested RPC
per managed parent is accepted at a time.

An explicit `AdmissionRejected` response means `execution_state:
not_executed` and `retryable: true`. The client has no hidden queue: one
`BrokerClient` admits one call at a time and rejects additional concurrent
calls. Retryability permits the caller to make a new decision; the runtime
never automatically replays scientific work. A lost connection after
submission remains `OutcomeUnknown`, even when overload is suspected.

The dispatch owner alone can clear its delegated state and release its
reservations. Resource admission, worker-lock admission and worker transport
waits share the request's absolute deadline; an earlier caller deadline takes
precedence. Cleanup and final ownership bookkeeping may still delay returning
after expiry, so this is not a hard bound on the method's total wall time.
A known application error is distinct from uncertain execution. An uncertain
CPU result fences its worker, while an independent healthy CPU worker remains
usable. Uncertain GPU work continues to fence its physical device until the
existing cleanup/ownership protocol permits release.

When an execute/finalize request expires or is cancelled before submission,
the preceding successful prepare/execute still owns its retained reservation.
No scientific work is automatically repeated. Confirmed cleanup releases its
ownership; uncertain cleanup retains reservations and artifacts. These broader
control-plane contracts require real socket/process saturation validation;
unit and CPU fixture checks are not pretrained GPU backend certification.

`BrokerClient.cancel(request_id)` uses the control lane without taking the
call lock. Its acknowledgement reports a cancellation request, not confirmed
resource release. A queued cancellation cannot stop another active owner.
After submission, the original dispatch owner stops and observes its worker
before releasing resources; a cancelled scientific outcome is not retried.

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
