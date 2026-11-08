# Model lifecycle and adaptive execution

## Qualified release boundary

The integrated runtime has local AF2 qualification on an RTX 3090: prepared
input saturation with one model owner, a complete 13-evaluation shared-model
Beam workflow, physical SC/BUNS/Holes scoring, and durable raw/Score recovery.
CPU workers are not concurrent GPU inference owners. That qualification does
not establish a new end-to-end speedup over the preceding optimized runtime.

Ordinary search preserves `serial` / `continuation` RNG semantics. The qualified
serial AF2 comparison uses `execution_mode=in_process`, resource mode `legacy`,
ipae/ipsae, and no pose-dependent position selector. In-process AF2 with a local
resident/adaptive resource broker is rejected before model construction; use
`legacy`, an external resource owner, or a daemon with local resident mode.

Persistent Beam with live PyRosetta poses requires explicit
`search.execution.mode=isolated` and fixed-request RNG. Shared-model physical
Score was exercised with both one and four CPU workers and one model owner.
Serial persistent Beam with live poses remains unsupported. Existing configs
are not silently converted between these scientific RNG protocols.

AF3 resident execution, ESMFold/ESMFold2, RNA, RFD4, BioEmu2 and other adapters
retain their individual qualification boundaries. Adapter registration and CPU
CI do not certify their model weights, credentials, geometry, multichain
behavior, ensembles, concurrency, or real GPU science for this release.
See [recovery compatibility](reproducibility.md#runtime-recovery-compatibility).

## Earlier AF3 integration

Integration note (2026-11-07): the frozen r38 AF3 resident increment is wired
to the r40 unified-kernel stages in this tree. Resident mode remains an
explicit experiment; the default process-pool/fixed-one/pickle path is
unchanged. CPU contract tests pass, but this integration has not had a new
GPU scientific-equivalence or throughput acceptance run.

The runtime schedules work underneath the existing scientific coordinator.
It does not select candidates, modify seeds, change sample counts or precision,
or decide whether a mutation is accepted. Independent legacy trajectories run
in separate processes because their global configuration and RNG are mutable.

`astrolabe.model_execution.KernelStages` separates proven CPU preparation,
device execution, and CPU finalization. Its loader runs once per private model
owner. The existing `NativeModelClient` transport keeps a model alive across
requests and transfers the scientific RNG state. Backends with a monolithic
vendor call retain that whole call as a conservative device operation.

Device execution includes reductions whose location affects scientific values.
The public ESMFold adapter requests its original masked atom-confidence mean
inside the model owner's device lease, before moving tensors to the host.
The coordinator receives those reduced values and retains the original NumPy
normalization and chain slicing. Raw `infer` and PDB interfaces remain available
with their original return shape. A complete workflow comparison detected that
performing this mean after host transfer changes selector statistics even when
coordinates, sequences and final RNG agree; kernel-output parity alone does
not qualify that consumer boundary.

The current implementation adds this contract to the existing evaluator and
mutator adapters. It uses their actual registered families and aliases; new
model families mentioned in the architecture proposal require their own
adapter and qualification.

## Opt-in AF3 cache hotfix and native beam ownership

Use the ordinary command with shared configuration and an already-valid
scientific case overlay; no caller-side `OwnedAF3` wrapper is needed:

```bash
ASTROLABE_RUNTIME_CONFIG=/absolute/path/to/runtime.json astrolabe run \
  -tinterface --standalone /absolute/path/to/common.yaml \
  --overlay /absolute/path/to/case.yaml \
  --overrides-file /absolute/path/to/Astrolabe/config/presets/af3_adaptive.overrides \
  --dry-run
```

Remove only `--dry-run` to execute. The preset selects a private owner and
enables adaptive policy; its concurrency/resource bounds remain those of
`protein_beam.service.adaptive`, not a recommended or inferred worker count.
The CLI applies post-overrides (including the preset) before the case overlay,
regardless of argument order. Keep execution/service settings out of the case
overlay so it cannot overwrite the preset or resource bounds. Put the caller's
qualified `max_concurrency` and memory budget in the shared `common.yaml`
configuration or explicit overrides; the case overlay must not override them.
An explicit `search.execution.socket_path: null` owns one AF3 context per
repeat. A nonempty path reuses the caller's external owner and never drains
it; adaptive external ownership is rejected. Missing execution fields are
errors. Without adaptive settings, the owned service is fixed1, unchanged.
The coordinator/workers hide CUDA; the owned service retains the original
GPU assignment. Owned lifecycle, stdout events and stderr diagnostics remain
under each repeat's `output/services`. Saved replay configuration retains
null ownership intent, so resume resolves a fresh private socket rather than
reusing an ended owner's transport. The beam checkpoint records the actual
socket separately. Adaptive plus an inherited `ASTROLABE_RESOURCE_SOCKET`
is rejected before loading; the launcher does not hide that broker.

`config/env/runtime.example.json` includes a version-1 `command: ["run"]`
profile template. Replace its paths with operator-provisioned paths. Its
`pid_namespace: false` keeps NVIDIA's host PID accounting and `/proc` in the
same namespace, which the adaptive sampler requires. Other profiles retain
the existing PID isolation unless they explicitly opt out; filesystem and
environment containment remain enabled.
Its
single-file read-only plugin bind and
`ASTROLABE_PJRT_CACHE_HOTFIX=jax-0.4.34-cuda12` select the cache hotfix
independently of adaptive policy: patch-only fixed1 is valid. For stock
fixed1, omit both this bind/marker and the adaptive preset. No binary is
downloaded, built, installed or shipped at a project-local default path.

The sole executable pin definition is
`astrolabe/runtime/pjrt_cache_hotfix.py`. It requires installed metadata for
`jax`, `jaxlib`, `jax-cuda12-plugin` and `jax-cuda12-pjrt`, all at `0.4.34`,
and hashes the final mounted/discovered `xla_cuda_plugin.so` against
`a2297ac0796d6961b24f9dc44131b529108ec684be421bb6a622efd1b6f87a25`.
Use the plugin destination from the selected image/interpreter's installed
distribution, not a guessed site-packages directory. Validation runs after
runtime selection and before CLI/JAX imports, and again before AF3 backend
loading. Unknown markers, incompatible metadata, discovery conflicts, missing
files and wrong hashes fail closed. There is no SM80-only device restriction:
the supplied binary's SM80 build label does not exclude the previously
qualified RTX 3090 (SM86). This does not qualify other hardware.

Native `run --dry-run` uses CPU-only container execution (no `--nv`) and
validates composed beam/service settings and any selected hotfix without
starting an owner, creating run artifacts or loading models. It establishes
configuration/package integrity, not GPU compatibility, scientific parity,
or throughput. The cache hotfix does not change numerical operations or
scientific parameters.

### Opt-in numeric AF3 transport

The owned service also accepts `protein_beam.service.concurrency_mode: resident`.
This experimental mode keeps one model process and one set of device weights.
The first request for each input signature loads an executable under a lock;
later requests reuse that in-memory executable with separate input arrays, RNG
keys and results. Increasing admission does not launch a process or load another
executable for a known signature. A new signature still requires compilation or
cache loading. The normal `process_pool` mode and fixed single inference remain
the defaults.

The pinned PJRT runtime uses one compute stream per device. Resident mode orders
host submissions through one launch lock, which ends when the executable returns
its asynchronous results. It does not wait there for device completion.
Preparation, pending completion and CPU extraction remain request-local.
`resident_executable_dispatch` records launch waiting separately from
`resident_executable_enqueued` host submission time. Multiple admitted requests
are a bounded queue ahead of that compute stream, not independent GPU streams.
Unordered concurrent calls into the same loaded executable failed exact AF3
output comparison; they are not a supported execution policy.

Resident mode uses the existing adaptive utilization, queued-demand and memory
checks when `service.adaptive.enabled: true`. With it disabled, only one model
inference is admitted, while other requests can prepare or finalize. Host
thread overlap does not itself prove GPU kernel overlap or higher throughput.
This mode requires the private owned service, rejects the resource broker and
legacy `--pipeline`, and retains normal shutdown/drain and device-failure
quarantine. Use numeric transport below to keep scientific object extraction
on the CPU caller.

`evaluator.alphafold3.service_transport: buffers` selects a same-host numeric
buffer boundary for the owned AF3 service and `RemoteModelRunner`. The default
remains `pickle`; inference concurrency remains fixed1 unless adaptive policy
is separately enabled. Select the same transport in the owner and clients.
This is an experimental transport, not a throughput qualification.

The CPU caller keeps the full native featurized batch. Only the output of
AF3's own `remove_invalidly_typed_feats` crosses to the model owner. The owner
runs the unchanged native upload/model/materialize stages, then publishes
raw `ModelResult` arrays. Native extraction runs in the caller, with its
original full batch. Output writing, scoring and candidate selection retain
their existing order and implementations.

Arrays use private node-local **tmpfs** mmap bundles under `/dev/shm`, independently
of the socket, cache and artifact filesystems. This is required, not a path
preference: NFS retains open-unlinked files as `.nfs` entries, which prevents
request-directory reclamation while scoring still holds mappings. A private
`buffers.json` beside the socket identifies the data namespace; children share
their parent's namespace without owning its cleanup. Both scientific endpoints
must see that same tmpfs mount. Separate containers need an explicit common
tmpfs bind; an unavailable namespace or exhausted tmpfs fails without filesystem
spill or pickle fallback.

JSON socket messages contain only
control fields and a request token. The adaptive parent neither maps arrays
nor imports their scientific classes. The existing pool holds admission until
native completion and output publication, not merely until an input is
accepted. No new scheduler, process lease service or arbitrary-object codec
is introduced. Contiguous arrays have a publication write, not a claim of
end-to-end zero-copy. Mapped outputs are writable copy-on-write views and
remain valid after their names are unlinked, including slices retained by
scoring consumers.

The new listener never unpickles or automatically falls back. Manual
`describe_snapshot`, `adaptive_control` and `shutdown` use
`inference_service.json_call`, not the legacy `call`. `--pipeline` is rejected
for this transport; its native kernel still preserves the real stage boundary.
The separate native-daemon backend is not migrated by this opt-in.

After a confirmed successful response and output mapping, the client removes
its request directory. On EOF, model error or a lost response, it retains
the directory and does not retry. Model failures quarantine the numeric owner.
Shutdown preserves nonempty buffer roots and records
`retained_buffer_root` in managed lifecycle state; process exit alone does
not prove a client consumed its outputs. These temporary retained directories
require explicit recovery, never automatic prediction replay.

This removes IPC reconstruction of `Structure` and `AtomLayout`, not XLA
executable-cache deserialization. The pinned cache hotfix above is independent
and remains required for that runtime. Persistent model owners retain their
loaded executable between requests. A newly expanded independent owner still
has startup and cache-load costs; this transport does not claim to remove them.

### Demand windows and forwarding cost

Adaptive expansion requires queued demand throughout the observed utilization
window. A low-utilization window that also contains starvation reports
`supply_limited_window` rather than adding a model process. Decision events
include `window_loaded`. `ready` counts complete evaluator RPCs waiting at the
pool gate, not prepared device inputs. Low average GPU utilization alone does
not establish that another model process will improve throughput.

The monitor sleeps when there are no active or queued requests. An idle
transition clears its sampling and throughput comparison window. Memory
headroom, telemetry freshness, warming and quarantine checks still apply.
The target utilization and concurrency defaults are unchanged.

The private adaptive beam path forwards an opaque scientific payload. The
proxy reads routing metadata and reply status, while the child and final
caller decode the scientific objects. This avoids loading NumPy and AF3 result
dependencies solely to relay them. Header/body identity checks, same-UID
transport, one forwarding attempt, error quarantine and shutdown draining
remain in force. Fixed1 and control calls retain their original transport.
These changes require real same-load end-to-end validation before any speedup
claim. Fixed1 remains the default.

## Split loading and reuse across sequential trajectories

`StagedLoader(prepare, place)` keeps CPU checkpoint reads, construction and
state installation outside the GPU load lease when its host-memory envelope
is declared. ESMFold uses this boundary in qualified profiles.
The prepared CPU weights retain a RAM reservation while placement waits.
Placement admits device memory and atomically transfers the prepared RAM
registration to the resident model; it does not request the same host memory
twice. Failure retains the appropriate owner reservations until actual exit.
Unconverted loaders and preparations with unknown host-memory requirements
keep their conservative original loading boundary.

ESMFold reports actual host wall/process times for checkpoint loading, model
construction, state installation and placement. Placement completes on the
device before readiness. Parameter/buffer device, dtype, count and logical
bytes are inspected after placement. Logical tensor bytes are distinct from
allocator/NVML memory, and host timings are distinct from GPU kernel time.

An explicit local pool can retain compatible owners across sequential public
workflow calls in a persistent worker's main thread:

```python
from astrolabe import Astrolabe
from astrolabe.runtime.model_pool import local_model_pool

# Attach this process to an independently owned resource_session broker first.
# Keep ASTROLABE_MODEL_OWNER=trajectory throughout this worker's lifetime.
with local_model_pool(max_models=1, backends={"esmfold"}) as pool:
    for cfg in existing_trajectory_inventory:
        Astrolabe(cfg).run()  # Existing reset, seed and scientific run boundaries.
```

Concurrent trajectories still need separate scientific processes. The pool is
opt-in and neither creates trajectories nor increases their sample budgets.
Borrowed handles release their attachment at run end; only the pool closes
the actual owner. Residency stays charged until the persistent process exits.
`max_models` bounds retained pool owners. An incompatible acquisition falls
back to the existing private path and can add a separate allocation, so a
bounded acceptance runner must reject unexpected misses explicitly.

Reuse matches code/assets/runtime/settings, broker and visibility scope,
the resource profile, and the complete initial scientific RNG state. A lazy
CUDA seed can match an initialized caller only with the actual first load's
captured CUDA generator state and visibility before placement. A hit restores
that load's observed post-initialization RNG state. It does not run the
initializer again. Unknown or invalid execution responses quarantine the pool;
the runtime does not retry them. The persistent acceptance runner additionally
binds warm acquisitions to the first completed cold worker receipt by digest,
PID, model identity and real checkpoint-load evidence.

The pool also retains the observed CPU RNG effects of ProDy's optional logging
import. The first import consumes NumPy RNG after trajectory seeding; a cached
import in a later trajectory otherwise skips that consumption. Replay requires
the same complete CPU RNG input and the same live module, and an unchanged CUDA
state across the original import. It restores only the saved CPU effects while
preserving the caller's actual CUDA state. This dependency boundary has its own
before/after evidence and hit count; the model's strict CUDA reuse proof still
applies. Dependency import, capture or restore failures quarantine the pool and
fail the trajectory, including errors raised after a partial state write.

`ASTROLABE_REQUEST_CONCURRENCY=0` makes individual native execute requests
exclusive under the shared broker. Other values retain the model profile's
upper bound and never grant concurrency beyond its qualification.

## Enable the existing CLI or API

The default remains `execution.mode=legacy` while qualification is incomplete.
An allocated device is required for adaptive execution:

```bash
CUDA_VISIBLE_DEVICES=0 astrolabe run --task my_task -- \
  execution.mode=adaptive
```

The same configuration is honored by `Astrolabe(cfg).run()` and the
`run_repeats` entrypoint. It can also be provided around direct model APIs:

```python
from astrolabe.model_execution import model_run_scope
from astrolabe.runtime.session import resource_session

with resource_session({
    "mode": "adaptive",
    "devices": "allocated",
    "cpu_slots": 2,
    "max_concurrency": 2,
    "telemetry_path": "/my/run/resource_events.jsonl",
}):
    with model_run_scope():
        # Existing model construction and scientific requests.
        run_existing_workflow()
```

`max_concurrency` is a deployment ceiling. The controller starts conservatively
and permits exploration only for explicitly qualified requests; a ceiling of
two does not immediately launch two kernels. `resident_serial` uses the same
resource broker with a compute ceiling of one.

The broker resolves physical GPU UUIDs and checks CUDA/Slurm visibility. Its
Unix socket is inherited as `ASTROLABE_RESOURCE_SOCKET`. Independent clients
sharing a GPU must attach to the same owner; competing owners fail rather
than independently assuming that the same free memory is available.

## Independent trajectories

`python -m astrolabe.runtime` supervises an explicit inventory of commands.
It creates no additional scientific tasks. Command identity, order, argv,
seeds and output namespaces are supplied by the caller:

```json
{
  "schema": "astrolabe.independent-commands/1",
  "execution": {
    "mode": "adaptive",
    "devices": "allocated",
    "cpu_slots": 2,
    "max_concurrency": 2
  },
  "max_trajectories": 2,
  "command_timeout_seconds": 600,
  "commands": [
    {
      "identity": "trajectory-A",
      "argv": ["python", "existing_workflow.py", "--seed", "17"],
      "output_dir": "/my/run/A",
      "cwd": "/my/project"
    },
    {
      "identity": "trajectory-B",
      "argv": ["python", "existing_workflow.py", "--seed", "18"],
      "output_dir": "/my/run/B",
      "cwd": "/my/project"
    }
  ]
}
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m astrolabe.runtime \
  --manifest commands.json --report command_report.json
```

Children use separate process groups. Nonzero exits are retained as failures;
a running command that times out is recorded as `outcome_unknown` and is not
replayed. Cleanup terminates the owned group and checks for live descendants
after its leader exits; a leader's return code alone is insufficient. Existing
nonempty output namespaces are protected. The supplied
scientific program continues to own its usual checkpoints and outputs.

## Resource and lifecycle evidence

GPU model residency and transient execution are separate reservations.
Unknown device subgraphs execute conservatively; certified requests can share
only within the physical device's resource and concurrency limits. CPU phases
release the GPU compute lease. A `Pending` result retains the lease and its
buffers until actual completion and host materialization.

Descriptions include `runtime_lifecycle` with PID, worker epoch, model identity,
loader count, declared residency, readiness, and available device facts.
Requests return a runtime receipt with their request identity and stage wait/run
times. Unknown weight/GPU-residency facts remain unknown. The legacy daemon
`model_load_count` field counts factory initialization; its explicit
`model_load_count_scope` prevents treating it as a checkpoint-read counter.

`resident_model`, `resident_bridge`, `one_shot`, and `unknown` are different
capabilities. A live bridge or a component cache does not establish that all
model weights remain on the GPU. Resident reservations are released only after
the private worker exits or the broker observes that its PID/start epoch ended.
Timeout and transport disconnect do not establish device completion.

The broker records GPU samples, queue/resource state and control decisions.
It distinguishes work supply from saturation and observes completion throughput
alongside utilization and memory. GPU utilization is a sampled activity measure,
not a proof of full compute efficiency.

Physical device concurrency starts at one, including across different model
identities. Probing requires certified ready work, a completion observation
window, activity samples below `target_utilization` and memory headroom.
`memory_pressure_fraction` gates growth and retracts the device limit; throughput
regression and OOM also retract it. These policy thresholds are tunable starting
values, not backend performance certificates. `controller_decision` events retain
the measured activity, throughput and reason. `activity_reason` distinguishes
running device work, queued admission, active CPU work and absent ready requests.
The last observation alone cannot establish whether missing work comes from
scientific causality, upstream supply or application I/O.

Native AF3 clients send the original seed to the model owner for PRNG key
construction and request native structure extraction there. Even enumerating
JAX CPU devices can initialize GPU backends in a visible-GPU process. Keeping
these operations in the admitted owner prevents an unaccounted coordinator CUDA
context from blocking subsequent conservative requests. The original JAX key
constructor and native structure/scoring function remain the scientific path.

## Model ownership inside an isolated trajectory

For explicit independent command inventories, `execution.model_owner=trajectory`
lets each already isolated trajectory hold its native kernels directly, removing
an extra model process and local IPC. A different process must own the resource
broker; using this option with an embedded same-process broker fails before load.
Requests must execute on the trajectory's main thread. Alternate Python,
provider, cwd, environment overrides and daemon socket attachment are refused.
Separate trajectory processes continue to isolate mutable scientific globals.

This option uses the same loader, RNG transfer contract, stages and GPU admission.
Settings, requests and results retain the daemon's bounded protocol-5
serialization boundary. Mutable input storage and kernel-reused output buffers
cannot alter the caller's input or its retained earlier results.
Logical shutdown drops the handler, but CUDA allocator/RSS reservations remain
charged until the trajectory actually exits. The supervising command timeout
terminates the process group with an unknown outcome, without replay. In-process
execution has no separate RPC timeout boundary. Use the daemon owner when its
separate environment or process fault boundary is needed. Default ownership
remains `daemon`.

## Qualification

Automatic resource profiles are selected by exact daemon identity through
`ASTROLABE_RESOURCE_PROFILES`. They must carry verified source and selected
weight SHA256 manifests, actual request shape hashes, physical GPU UUID and
driver, Python ABI and framework package versions. Explicit checkpoint paths
also contribute their contents to model identity, using a digest cache keyed
by filesystem change identity. A changed request shape falls back to unknown
memory and exclusive execution. A certificate for another visible GPU cannot
authorize the selected model device. Production certificates across interpreters
or modified provider environments require child runtime attestation and are
currently refused. Directly supplied qualification profiles
permit bounded experiments; they are not automatically enabled for production.
Profiles must be scoped to the tested model, input shapes, runtime and hardware.
Conflicting explicit `device`/`gpu_device` declarations, including declarations
in different constructor/configuration scopes, are refused for certification.

The session resolves a finite VRAM envelope from current allocated-device free
memory, or accepts explicit per-device `vram_bytes`. Live samples still constrain
each admission. CPU slots must fit process affinity. The default host RAM envelope
is the minimum of available RAM, readable cgroup v2 headroom and declared Slurm
memory. Explicit `resident_ram_bytes` transfers atomically from initialization;
`scratch_ram_bytes` covers transient work. For undeclared residency, the enabled
session observes worker RSS by PID/start epoch, retains its high water mark and
keeps it charged until process exit, including allocator caches after unregister.
Unknown initialization is serialized. This observation does not predict unknown
transient peaks or enforce a memory cgroup; models without measured envelopes
remain unqualified for resource-safe concurrent execution.

A private daemon drains accepted requests, wakes its listening socket and waits
for the actual process exit before acknowledging resource release. A drained RPC
response alone does not establish exit; the configured shutdown timeout permits
framework teardown before TERM/KILL escalation. Exit status and escalation remain
distinct lifecycle evidence.

A GPU exception under broker admission quarantines its worker and retains the
lease fence until completion is positively established or the worker exits.
The legacy path without a broker keeps its existing recoverable-error behavior;
OOM still quarantines. A broker shutdown timeout retains its socket, GPU UUID
locks and monitor. Persistent PID/start-epoch records prevent a replacement broker
from treating surviving orphan workers as free capacity after a broker crash.

CPU registration counts model ownership even without GPU residency. Descriptions
of lazy component caches must be interpreted with the producer metadata:
Chai's `weights_loaded` means at least one component is cached, while its
`cached_components` gives the count. Neither that fact nor a live bridge proves
that the complete model remains on the GPU. Retrieve a fresh `describe` after
requests to inspect dynamic Boltz/Chai/AF3 state.

The real-model harness is:

```bash
python scripts/validation/adaptive_runtime_acceptance.py \
  --output /my/evidence/qualification \
  --weights /models/ligandmpnn/proteinmpnn_v_48_020.pt \
  --input examples/input/9TP_cid.cif --chain A
```

The same fixed inventory can qualify trajectory ownership using
`--model-owner trajectory` and a separate evidence directory. The existing
daemon comparison must remain retained; changing transport cannot reuse its
timings as if they measured the same implementation.

It retains complete outputs, raw arrays, RNG boundary hashes, resource events,
fixed task counts and interleaved direct/resident-serial/adaptive measurements.
It fails on output/RNG mismatch or source drift and makes no automatic speedup
claim. A separate worker invocation with `--profile` records a PyTorch CUDA/CPU
trace; profiler overhead is excluded from the timed comparison.

CPU contracts and vendor stand-ins verify scheduling mechanics. Real model
qualification remains specific to the tested weights, shape, precision, seed
inventory and hardware. An unavailable dependency or model is reported
separately from a passed backend; qualification of one model does not certify
the others.
