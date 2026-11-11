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
