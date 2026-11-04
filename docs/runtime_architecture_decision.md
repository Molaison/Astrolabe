# Runtime ownership and execution decision

## Decision

Improve the existing process-owned pipeline at the boundaries where evidence shows a defect or avoidable work. Keep trajectory state, proposal RNG, scientific acceptance and stage commit in their existing owner. The first pilot validates experimental worker completion identity. Retain resource monitoring when reducing CPU startup cost. Move only audited host-only backend work outside device admission.

The source under review is `d66ee714f2e755948534f95d9d125287ce8dbd91`. The old Mental Model is based on `b537b8a`; current runtime claims below come from the branch source and reproduced probes. Main's later observation/ownership changes were inspected as prior art, not merged into this branch. In particular, the candidate pool source is identical at the target head and `main@2a8b72a`.

## Grounding and owners

| Boundary | Existing owner | Data and invariant |
| --- | --- | --- |
| Repeat scheduling | `run.run_repeats`, `runtime.session` | Unresolved config is cloned per repeat; explicit seeds increment by repeat index; independent repeats use process isolation |
| Scientific state | `Protomer`/shared `Segment`, `Decisions`, stage transaction | A trajectory's next mutation follows its previous evaluation/decision; accepted mutation and stage-entry commit are different boundaries |
| Request identity | `execution_identity.scientific_operation` | Read-only owner/input snapshot, accepted revision, stage/gate and file fingerprints; this context is not a second trajectory state store |
| Resident models and resources | `BrokerSession`, broker `State`, `ResourceLedger` and guardians | Process/model identity, bounded pending bytes, CPU/RAM/VRAM admission and confirmed release; an unknown outcome cannot be replayed as fresh work |
| Backend representation | Each evaluator/worker and its native model | Prepare the exact native input, execute device operations under admission, finish detached host outputs outside it when audited |
| Experimental pool result | `CandidatePoolScheduler` | Submitted `(candidate_id, generation)` is authoritative; a returned result cannot choose its own identity |

Caller-first sketch for the selected design retains the existing surface:

```python
pool.submit_generation(generation, tasks)
collection = pool.collect_until_ready(target_generation=generation)
# A successful collection contains only results matching their submitted task.
```

The ordinary MCMC path remains a blocking call stack inside one trajectory owner. Process overlap between independent repeats is already implemented; it is not newly credited to this change. Experimental best-of-first-completions selection is a distinct scientific protocol and cannot substitute for serial MCMC in a performance comparison.

## Alternatives considered

| Architecture | Benefit | Cost and decision |
| --- | --- | --- |
| Strengthen existing process-owned pipeline | Reuses repeat seeds/reset, resident workers, resource ledger, request fences and scientific writers; small reproducible changes | Selected. Live pipeline overlap and model throughput still require measurement |
| Versioned proposal queue plus trajectory actor | Explicit work IDs, parent/stage epochs, cancellation and durable commit ledger | Deferred. Requires complete RNG/checkpoint and OutcomeUnknown semantics; introducing another state writer before proving these would increase correctness risk |
| Persistent generic trajectory worker pool | Amortizes imports across jobs | Deferred. Arbitrary plugin/global/framework state is not certified to reset safely |

Two narrower R1 implementations were compared. An in-process direct fast path is smaller, but `execution.resources={}` still means allocation-derived limits. It would silently remove monitoring, so it is rejected. A monitored child that reuses the existing serial lifecycle could amortize cheap CPU repeats while preserving resource ownership. Eager import reduction is a structurally different alternative, but needs a fresh import profile; the app loader is already lazy. Neither idea earns a speed claim from source inspection.

For the ESMFold1 boundary, preserving the native audited atom conversion and only detaching the host fields needed by PDB formatting is preferable to copying every output and running the whole native helper on CPU. A generic phase framework would duplicate existing lifecycle machinery. Any native-helper split must preserve atom/residue ordering, chain IDs, coordinates, B factors, score arrays and RNG, with a conservative path for unknown implementations.

## Measured selection evidence

Agent C's injected replies and the lead's independent run both show that baseline `CandidatePoolScheduler` can select a foreign candidate and a never-submitted future generation. The positive control still selects the expected task. This is a real defect in the experimental boundary under an erroneous worker reply; it is not a claim of a spontaneous error in default MCMC. The pilot validates identity before the result enters the ready set and uses existing failure handling under the original submitted identity.

Agent A decomposed archived W0 traces. The serial resident median was 32.418 seconds, with 6.335 seconds before broker readiness and roughly 8.143 seconds per trajectory process; the three CPU evaluator stages together occupied only milliseconds. This locates the regression outside device work but does not identify a specific expensive import. The old direct median was 9.691 seconds. These are archived measurements on a different environment, not this branch's before/after speedup.

Agent B's boundary probe executes the actual worker function and pinned native helper with injected tensors, inference and formatter hooks. It shows CPU PDB formatting inside the device lease and copying an unused output. That probe is contract evidence, not a pretrained model or GPU benchmark.

## Integration and review contract

One writer owns each module. The lead owns this decision, benchmark contract, integration, publication and independent reproductions. Agent A owns runtime/performance and the comparative CPU runner. Agent B owns the audited evaluator split and backend certification. Agent C owns experimental result identity and producer/scientific checks. Pilot acceptance precedes parallel product development. Cross-review must reproduce failures at the stated source tree; unsupported concerns stay hypotheses.

No scientific configuration, repeat budget, model precision, sample/recycle count, acceptance rule or stage semantics may be changed to improve timing. No speculative same-trajectory proposals are added. Existing same-device forward limits remain conservative until a real backend earns a higher setting. Incremental caching requires dependency and stale-cache negative controls; no broad cache is introduced from a name or registry entry.

The validation limits and frozen workload are in `runtime_benchmark_contract.md`. Backend facts must distinguish implementation, test-double contract checks, real CPU execution and real pretrained GPU execution. Missing checkpoint, allocation or process capability remains BLOCKED. No full recovery or GPU throughput certification follows from the pilot.

## Independently reproduced boundary repairs

The pilot was followed by complete finite-generation validation and a scheduler-lifetime `(generation, candidate_id)` admission set. Invalid or duplicate plans cannot dispatch a valid prefix; collection or failure cannot erase the replay guard. The existing production caller already constructs finite, fixed-parent lists. This is an in-memory single-caller boundary, not durable exactly-once execution. A capacity-one blocking-submit prototype was rejected: with the same three tasks it changed the first selected candidate from ID 0 to ID 1. A bounded asynchronous producer remains a separate unfinished design problem.

Broker admission now charges pending-memory wait, compute wait, worker-lock wait and entry bookkeeping to one deadline. An expired request cannot receive a fresh dispatch budget. The lead reproduced a one-second request dispatching after 1.301 seconds before the repair and timing out without dispatch after it. Cross-review also found that a timed-out queued request could clear another active request's delegation, causing that active request to fail. Only the worker-lock owner now clears that state. The fixes retain existing request-ID consumption, OutcomeUnknown fencing and cleanup. Blocking bookkeeping and cleanup can still delay return after the timeout; this is an admission deadline, not a hard real-time return guarantee.

The ESMFold-specific split preserves audited native atom conversion and required host materialization inside device admission, then formats detached CPU outputs outside it. Exact native CPU differential tests and worker-boundary negative controls verify this representation/lifetime contract. Requested host payload fell from 635,728 to 20,648 bytes and from 34,256,928 to 326,672 bytes for the two synthetic output shapes. Those counts are not measured CUDA transfers. Independent CPU total timings include regressions, so neither CPU nor GPU throughput acceleration is certified by this change.

The first actual same-runner CPU comparison of the correctness changes completed 30/30 trials and all scientific/resource-release gates. Candidate median direct/resident-serial/resident-adaptive times were 2.8286/7.2894/7.3085 seconds; frozen d66 baseline times were 2.7900/7.3248/7.3731 seconds. Paired ranges overlap. This is retained as a negative performance result and a clean starting point for a separate R1 import-cost hypothesis. A narrower deferred-SciPy import candidate can preserve the existing owner/process/stop architecture; it must earn its own end-to-end result before a performance keep decision.


## f102 continuation: recover existing owners, preserve scientific boundaries

The server inspection recovered unmerged `cloud-20261103-fix` and `opt` linked
worktrees. Their complete tracked/untracked snapshots were preserved before reuse;
these original directories are not independent clones and were not reset or edited.
The selected implementation incorporates the already written absolute wire
deadlines, separate business/control/nested ingress lanes, cancellation,
CPU-worker/physical-GPU failure isolation and retained prepared-resource ownership.
A root-reproduced waiting-count leak after actual compute admission was then fixed
by tracking the transition actually counted, rather than clearing a guessed state.
The resulting source passed 309 real Linux process/AF_UNIX contracts, zero skips.
This broad correctness integration alone did not earn an additional speedup:
the full same-host 30-trial comparison was noisy/negative and is preserved.

The next performance pilot reuses the old opt lazy PyRosetta loader and source
scanner; it retains previously verified SciPy deferral. The native import set is
published atomically under the existing first-use lock, and six direct helper
entry points require it before access. Loss formula bodies/registry signatures
and all scientific settings remain unchanged. Installed-PyRosetta controls and
complete W0 passed independently; the five paired comparisons improve every
mode beyond the 20% CPU engineering target. This does not make resident the
fastest route for short CPU work, and is not a GPU speed claim.

No new general actor/scheduler, speculative MCMC proposal, durable commit store
or dependency cache was added. Generation prevalidation remains finite and
scheduler-local; OutcomeUnknown fences work without claiming crash-persistent
exactly-once semantics. The proposed stream-wide lineage/backpressure/ordered
commit architecture and GPU overlap are still incomplete.

The real pretrained pilot entry is a narrow public ProteinMPNN runner control,
with identical native inputs/settings for legacy and private-resident execution.
Both modes retain their model; there is no artificial reload baseline. It cannot
certify evaluator W1, full registry selector/acceptance, other variants, shared
broker fairness or OOM recovery. One separately owned Slurm allocation is required;
its independently resolved GPU UUID is bound before CUDA initialization. Resource
metadata/adversarial CPU tests only validate the launcher contract. Actual device,
terminal accounting and every request's scientific/lifecycle evidence must come
from the recorded GPU run. Frozen source remains in an immutable worktree; result
reports are added on a separate evidence branch.
