# Architecture convergence implementation ledger

This is in-place migration on baseline `b537b8a`, not a second engine. Public CLI, Hydra/Python entry points, family registries, search rules and CSV consumers remain the compatibility boundary. No end-to-end model or GPU test was run for this change.

| Area | Implemented | Limit |
|---|---|---|
| Registered state | RuntimeState owns constants/variables; RuntimeContext binds a store; facade has no per-field mirrors. | Legacy globals and native backend state are not independent concurrent trajectories. |
| Mutation decision | One commit helper and ordered attempt/revision/assessment observations. | Existing policy, RNG and stage transactions remain authoritative. |
| Evidence | Isolated input dispatch; immutable request observations; clone/filter/merge provenance; unknown sample groups. | Not a complete native prediction-batch or chemical-state archive; do not infer sample identity. |
| Structural derivation | Rosetta hook records source-to-derived paths and operation. | Other processors are not automatically assigned invented provenance; confidence belongs to the original prediction. |
| Lifecycle | Deduplicated explicit shutdown/close, RF3 and peptide runner delegation, factory-failure cleanup. | Backend-internal construction failures and full device-memory release require backend-specific verification. |
| CPU overlap | Serial evaluator calls, ordered finalization, backpressure, propagated context. | No GPU scheduler or changed search policy. Final results remain retained; not a total-memory bound. |
| Persistence | Atomic compatible CSV/provenance plus additive terminal execution audit. | No cross-file transaction, power-loss durability or exact checkpoint/resume. |
| CPU dependency | NumPy confidence losses register without importing Torch. | Real tensor/device paths need their own environment evidence. |
| AF3 compilation reuse | Immutable 40-token policy at native tokenisation; inclusive preferred 256/512/1024 windows; JAX disk-cache setup/reset and a persistent host mount for native containers. | Explicit old bucket lists retain nearby/exact behavior. No output cache or same-result promise across different padding; JAX 0.4.34/cuDNN still needs its qualified PJRT serialization fix. |
| Module profiling | Opt-in shared configuration, spans across execution owners, process/resource sampling and offline HTML/JSON/Perfetto reports. | Diagnostic host spans and device samples do not prove kernel parallelism, scientific equivalence or performance improvement; original scientific and cleanup owners remain authoritative. |
| Fresh root raw import | Explicit `root_prediction_receipt`, original worker validation, new raw/import/Score evidence; root inference count zero. | Not checkpoint migration; fresh isolated fixed-request root only, unchanged prediction inputs/assets. Child predictions and old unknown fences unchanged. |

## Compatibility and recovery

`scores.csv` keeps its columns and multiple-assessment-per-step rows. Its existing full-history rewrite is now an atomic single-file replacement. The `incremental` argument remains accepted. The writer preserves destination permissions and symlink targets; serialization/replace failure preserves the old file and removes the temporary output.

`execution_audit.json` schema 1 separates invocation outcome, assessment acceptance, observed stage transitions, cached origins and observational errors. IDs are run-scoped, unknown native sample relationships are not guessed, and large array contents are explicitly omitted from this compact file. It declares `exact_resume: false`. Terminal publication errors are logged without changing scientific decisions; this file is neither a live checkpoint nor an all-or-nothing transaction with CSV/provenance.

Explicit contexts stay bound across a facade reset. Other legacy readers still use the active process-wide store. Migration must replace a writer and its consumers as one owned slice rather than maintaining two writable representations.

Normal direct stage rejection keeps the previously accepted mutation while restoring source-stage assessment. Consistency-gate seed/active attempts have their own boundaries. Rejected mutation attempts keep RNG consumption and can contribute to prefix early stopping.

The overlap default bounds unconsumed finalizations to `max_workers + 1`. Success results preserve request order. Errors are now surfaced in request order and backpressure stops later submission after an observed failure; this is an explicit error-handling improvement, not a claim of identical simultaneous-error precedence to the old unbounded producer loop.

## Evidence levels

Mol-1 is a narrow native-family adapter port (source `9e85ad16`, separate base/Constraint weights), not a legacy runtime merge. Auto selects from merged request constraints and dynamic hints, freezes both switchable assets, and replaces the current native client only after normal shutdown. Constraint-only legacy configuration remains supported. Real deterministic base → Constraint → base requests on the main integration tree matched their native atom coordinates/confidence exactly and stopped the previous process at each switch; 13 selection/input routes and 15 existing checks passed. Non-deterministic comparisons are retained as failures. Full recovery/service/performance qualification and daily promotion remain pending ([usage](mol1_constraint.md)).

| Evidence | Interpretation |
|---|---|
| Pinned baseline | Original commit, supported Python and targeted failures recorded before implementation. |
| Real-class CPU checks | State isolation, decisions, input/evidence transformations, lifecycle, scheduling and atomic output. |
| Differential replay | Baseline/current decision outcome, CSV, segment operations and next RNG draw compared; no native inference equivalence implied. |
| Real consumer checks | Existing CLI/config and paper-score consumers exercised without predictor execution. |
| Backend lifecycle methods | Real adapter methods with controlled owned runners, without model loading. |
| Native model / GPU / end-to-end | Not run under the task constraint; no scientific-success, numerical-equivalence or GPU-speedup claim. |
| Other feature branches | Not merged or treated as verified main capabilities. |

Use supported Python plus the repository's dev dependencies, `CUDA_VISIBLE_DEVICES=''`, and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Focused regressions cover runtime context, decision/stage transactions, pre-update, score normalization, structure initialization, dry-run configs and paper-score consumers. New boundary checks cover assessments, evidence, runtime support, backend lifecycle, execution observations and optional Torch behavior. Exact commands and final counts are recorded in the delivery report/PR, not implied by a test file's existence.

Broader backend migration, native sample reconstruction, exact recovery and adaptive GPU scheduling remain separately qualified capabilities. Lack of end-to-end verification must not be hidden by green local checks.

### Proton/pH main-interface port

The Proton joint mutator, lazy external runner and pH losses now use existing
factory/default/startup/resource owners. Only original product code and its
unit/control-flow checks are in scope; model/GPU/parity/release qualification is
owned separately. Historical pH branch reports do not establish current-main
acceptance. See [Proton/pH guide](ph_sensitive_design.md) for the consumer contract.

### Candidate identity extraction

The current `design.collection.CompletionPool` owns whole-generation validation,
pool-lifetime job ID uniqueness and submitted-job attribution of worker results.
The candidate-named facade continues to alias that owner. Restored jobs require
terminal admission proof to reconcile without execution; unknown work cannot be
resubmitted through this guard. Selection, budgets and RNG remain unchanged.
Deterministic identity regressions and the existing candidate/recovery checks
are included in the release CPU suite. This is a correctness fix, with no model
or performance qualification claim.

The legacy Mol-1 base adapter remains on the independent
[`research/mol1-qualification-20261108` branch](https://github.com/Molaison/Astrolabe/tree/research/mol1-qualification-20261108);
its [qualification contract](https://github.com/Molaison/Astrolabe/blob/research/mol1-qualification-20261108/docs/mol1-qualification.md)
separates historical native canaries from the outstanding current-main gates.
That preservation branch includes old runtime dependencies and is not intended
for a whole-tree merge.

### P0 request, cleanup and recovery boundaries

DPLM/RAPiDock no longer replay an uncertain scientific request or draw another
seed after transport failure. Shared stdio transport binds the response ID and
bounds partial writes/lines when a request timeout is configured. Socket
response serialization and identity failures remain unknown after send.
Collection preserves unknown through wrapped exceptions, leaving reconciliation
and budget accounting with Runtime.

ResourceOwner and StartupScope now publish cleanup receipts instead of marking
an attempted close successful. Failed handles survive; explicit cleanup retry
visits failed children without closing successful children again. Native
refinement, NA-MPNN and stdio process adapters retain their inner owners until
teardown succeeds. The CLI reports unresolved cleanup, preserves raw data, and
keeps the original scientific exception when both operations fail.

Release CPU coverage includes real process disconnect/malformed/identity/
timeout injection, post-compute publication failure, cleanup/reentrancy faults,
durable unknown accounting and raw/Score restart recovery. No scientific
selection, mutation/stage commit, RNG policy or scoring algorithm was changed.
The raw optional `candidate_scores` assertion now requires `None`, matching the
existing data class; this is not permission to persist finalized scores in raw.
Cross-version state migration, DPLM proposal-cursor persistence and lifecycle
qualification of every backend remain outside this P0 change.

### Scientific state, initialization and remaining boundary repairs

The subsequent P1 integration binds live restoration to one Runtime run and
stages restore data before mutation. Pipeline and lazy model initialization
retain failures, distinguish proof from residency, and reject ordinary replay.
Loss extraction and startup validation use configuration revisions rather than
object identity alone. Native PyRosetta initialization is process-owned;
MSA loading is atomic; provider pollution survives reset.

DPLM/pore/RFD4 expose validated scientific projections. Configured refinement
stores them with the existing Runtime at the proposal boundary and restores
before the next mutation, without rolling back rejection consumption or RNG.
Shared DPLM response caches and incomplete predocker sidecars are rejected by
the durable profile. Legacy live graphs, arbitrary stages/poses and unsupported
cross-version records are not upgraded by these adapters.

Score memoization now requires the native raw content identity, including array
bytes and referenced structure contents, as well as run/configuration bindings.
It returns caller-owned mappings; values without a full codec bypass reuse.
Masked arrays and unencoded array metadata cannot silently lose state in a
native snapshot. Lazy native owners reject access until readiness is bound,
including the construction-to-registration interval.

The optional initial DEBUG cache has a separate stateless evaluator contract.
It binds source/config/input/global RNG and verifies raw output contents before
reuse; evaluation that advances global RNG cannot publish a reusable entry.
Hooks and finalization run on both paths. Native backends without this contract
and legacy unbound pickles are excluded. Cache write/cleanup errors remain
observational and cannot turn a successful model operation into a retry.

Posthoc PARTIAL/COMPLETE receipts, derivation reuse and records now commit
together. Final content manifests identify filter/export/archive publication.
Cleanup and publication failures preserve original errors/evidence. Remaining
backend cleanup, shared Proton ownership, signal reentry, installation
completion and download-stamp guards are repaired; the sixth reversible Proton
patch records fork-pool and Squashfs cleanup changes. Unused vendor training
protocols remain outside the application release.

Release CPU checks include the new protocol suites and a configured-refinement
cut after proposal terminal publication: the resumed next proposal matches the
uninterrupted baseline. Native scientific canary, CI and deployed commit
identities are recorded separately in release evidence.

## Native vdM and normalized pH metric integration

The qualified vdM pair proposal, role projection and local Proton adapter are
integrated through the existing native mutator factory rather than restoring
the historical base/runtime wholesale. Fixed/target constraints, starting-root
site identity, proposal budgets and current ResourceOwner/seed boundaries remain
owned by their existing components. Frozen historical runs are not migrated.

The raw `ph_binding_endpoint` loss supplies local three-system Proton-H,
Rosetta and fa_elec S/P descriptors. Shared normalization/ensemble expressions
use the existing LossEngine; they do not introduce a second optimization loop.
Saved float32 Proton tables are promoted before changed-entry subtraction.
Old/new Rosetta readouts agree on the same saved poses; serialized-pose versus
original in-memory differences are retained separately. These checks do not
establish live model/relaxation parity or experimentally calibrated pH release.

### Opt-in cooperative refinement pipeline (r58 validated)

The configured native owner shares Runtime's durable call lifecycle with a
main-thread continuation. It freezes LigandMPNN daemon requests before leaving
legacy activation and moves transport wait only, preserving ordered application
and private intermediate RNG. Unsupported stacks and main-thread-only local or
borrowed owners retain explicit synchronous fallback. Score waiting+running and
prediction/raw-holding use independent bounded lanes; completed unpublished
results still occupy a fixed reorder window. Recovery rebases occupancy via a
credit epoch without replaying unknown inference. All three execution toggles
remain off by default. CPU checks and six native D5056_D15 AF3/Rosetta runs
passed exact scientific and continuation-RNG comparison. The profiled candidate
used staged proposals 12/12 and overlapped main control persistence with remote
wait; Score concurrency remained one. Two unprofiled replicates per version
averaged 347.722 s before and 339.769 s after (2.29% observed reduction).
This fell below the predeclared 3% benefit threshold; it is not a confirmed
throughput improvement or qualification for other workloads.


## 2026-11-10: configured budget stop and read-only finalization

The TNFa p10/p14 run produced 66 complete Scores, but the configured builder
forced `cap_error=True` and did not expose a generation stop. The builder now
uses normal proposal-cap termination and accepts explicit
`search.max_generations` (root generation zero); `steps` is not reinterpreted.

An explicit native CLI budget-snapshot export validates a terminal/drained old
journal and completed measurement evidence, then writes a separate result.
It neither bypasses source-bound resume contracts nor writes old checkpoints,
receipts or failed provenance. The historical generation-13 overrun remains
reported. Scope: CPU/control-flow checks and read-only export from the actual
retained run; no new AF3/Proton inference or pH experimental validation.

## Physical score backends and incremental preparation (2026-11-17)

This change started at `09f0460feb733b4675923ee580c977c7f7423e13` and was
rebased onto `38bf2d4497ac383754ff0aeeff435114e63fda38` before integration
verification. Three isolated implementation worktrees covered independent
scores, loss/selector contracts and AF3 preparation. The integrator owned
runtime dependencies, Boltz asset reuse and the final public entrypoint checks.

### Implementation and scientific boundary

SC uses [protein-interface](https://github.com/aarteixeira/protein-interface)
0.1.3 with its indexed parallel SC implementation. CMS uses
[py-contact-ms](https://github.com/ullahsamee/py_contact_ms) 0.1.1 to generate
Connolly SES dots; an exact KD-tree replaces its dense nearest-dot reduction.
Area/distance arrays are reused across distance weights. A real upstream API
comparison at weights 0, 0.5 and 1.2 passed at relative tolerance 1e-12 and
absolute tolerance 1e-9. This establishes equivalence to that pinned Python
implementation, not to Rosetta's numerical results.

Polar scores combine [Biotite Baker–Hubbard geometry](https://www.biotite-python.org/latest/apidoc/biotite.structure.hbond.html),
RDKit BaseFeatures donor/acceptor typing and Hydride H placement. Only Donor and
Acceptor feature families are queried; an all-family oracle verifies identical
atom sets. Full chemical context remains present for burial and intrachain
satisfaction. Incomplete explicit H, missing necessary CCD heavy atoms and
unsupported chemistry fail explicitly. These are geometric definitions, so
Rosetta energy/burial thresholds need recalibration after a backend change.

Exact [RosettaHoles2](https://pmc.ncbi.nlm.nih.gov/articles/PMC2998733/) depends
on DAlphaBall volumes and trained atom-type weights. The search did not establish
an existing standalone implementation with the same definition. Holes therefore
retains Rosetta; no cavity-volume substitute was introduced. Local/interface
selectors restrict the reported atoms while keeping their original environment.
The whole-chain deletion metric remains whole-chain.

`loss_config.py` owns deterministic backend selection, scope normalization and
selected dependency projection. `losses/selection.py` maps the existing
positional-selector language onto chain-relative residue ordinals, including
insertion codes. Native selector dependencies still request native poses.
Unsupported options, conflicting aliases and nonfinite controls fail before
evaluation. Existing RFD4 and multidomain recipes with native energy/radii
controls now explicitly select Rosetta. See the [migration guide](user_guide.md#61-物理评分-backend-与位置选择).

AF3 reuses unchanged MSA homologue encodings and conservatively keyed template
features. Sequence/seed-dependent native features and complete model batches
remain fresh. The evaluator owns a bounded cache with detached return values.
Custom CCD merges are request-local; their native reference cache is disabled
inside the existing scoped environment because its key includes object IDs.
Boltz similarly retains one CCD asset per runner with request-local molecule
copies and explicit shutdown cleanup. Protenix and Mol-1 reuse unchanged input
assets. See [preparation controls](adaptive_runtime.md#af3-deterministic-preparation-reuse).

### Retained review findings

Independent reviewers reproduced and closed scoped aliases losing their shared
structure preparation; partial explicit H silently losing donors; implicit
active losses missing dependency checks; stage entrance dependencies being
filtered by the destination's whitelist; and Boltz shutdown retaining its new
CCD cache. The integrator also fixed eager PyRosetta pose creation despite an
independent backend, nested disabled AF3 scopes inheriting the outer cache, and
native custom CCD contamination. Public LossEngine evaluation now exercises
all four independent score families without importing PyRosetta.

### Performance evidence

[Recorded observations](../examples/benchmarks/results/loss_af3_20261117/)
include complete samples, package versions and reproduction commands.
All physical timings use one 216-residue, 1,656-heavy-atom fixture, A versus B+C,
with 20 visible CPUs. SC uses the library's parallel pool. CMS uses
distance_weight=0.5 on both sides. Native pose and selector residue counts
were checked after serializing the same coordinates to PDB; the installed
PyRosetta reader otherwise imported this minimal mmCIF as an empty pose.

| Operation | Baseline median | Changed median | Interpretation |
|---|---:|---:|---|
| SC | Rosetta default radii: 6.073 s | Independent: 0.203 s | About 30× on this fixture; different radii/results |
| CMS | Rosetta default radii: 1.677 s | Independent: 8.909 s | Independent remains about 5.3× slower per new surface |
| H-bond count | Native public wrapper: 6.649 s | Prepared context: 0.000359 s | Reuse time; add preparation for a new structure |
| BUNS | Native: 0.616 s | Prepared context: 0.0517 s | Definitions differ; no numerical equivalence claim |
| Full AF3 preparation, fixed seed | 141.4 ms | 107.4 ms | 24% reduction in the recorded workload |
| Full AF3 preparation, changing seed | 443.3 ms | 406.8 ms | 8% reduction in the first run |
| Changing-seed independent replay | 522.1 ms | 492.9 ms | 6% reduction; host timing variation is visible |

Each paired run has five measured iterations after warmup. Polar preparation
for a fresh structure after library loading took a median 1.757 s in three
additional observations; the first H-bond call including preparation took
3.236 s. The small prepared-context H-bond time must not be presented as the
cost of a new structure. Independent CMS still spends most of its time in
upstream Python SES generation; its profile measured 10.041 s there and
2.675 s in the original dense reduction. The new reduction and alias reuse do
not remove the SES cost. Explicit Rosetta remains the faster CMS choice here.

Each full AF3 run uses 80+80 residues and 1,024 MSA rows with three-residue query
changes. Current base.yaml preparation settings are enabled and existing native
caches are isolated between the two sides. All 295 numeric feature arrays per
run, plus object metadata, matched exactly. No inference or GPU benchmark is
claimed. The pinned Python source is
`4b154d922759c918dd4b5a4d5a779853100fd20e`; its installed C++ extension is older.
These complete-preparation runs contain no templates. Real template parity
(including entity merge/split, mapping and coordinate changes) was separately
executed with a matched native Python/C++ installation. The template method's
AST is identical to the pinned source; matching-current-build template
qualification remains outstanding.

### Verification

The final targeted suite passed all 427 tests with zero skips/failures; nine
non-failing warnings were reported. The results and exact pytest file list are in
[verification.json](../examples/benchmarks/results/loss_af3_20261117/verification.json).
The retained checks cover public scoring, backend-aware startup and runtime pose
gating, sparse/insertion-code selectors, scoped preparation sharing, native
Boltz parser equality, AF3 invalidation/chemistry isolation and real template
features. The formerly external Holes fixture now uses the repository's C3
structure in an isolated native process, including selected-atom sum parity.
All 39 physical loss definitions in existing config/example YAML files also
passed parameter preflight after the two native recipe migrations.

The runtime dependency lock was regenerated with the repository's Python 3.11,
manylinux x86_64 hash-generation command. This is CPU/control-flow and native
feature/scoring evidence. It is not a full container rebuild, GPU inference
qualification, or a benchmark of complete Boltz/Protenix/Mol-1 preparation.
