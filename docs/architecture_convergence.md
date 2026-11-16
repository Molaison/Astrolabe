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

### Independent structural losses and AF3 preparation reuse (2026-11-16)

Implemented against `09f0460feb733b4675923ee580c977c7f7423e13`, with three
isolated implementation worktrees, one integrator and executed cross-review
reproductions following the poteto workflow. No change to mutation acceptance,
attempt budgets, RNG consumption or stage commit rules was intended.

| Requested outcome | Implemented boundary | Qualification |
|---|---|---|
| Reduce Rosetta scoring overhead | Independent SC, CMS, geometric HB and interface BUNS; `auto` / `independent` / `rosetta`; backend-sensitive Pose/package/DAlphaBall acquisition. | Real independent libraries on CPU, including official Python 3.11 / NumPy 1.26.4 / Biotite 1.5.0 pins. RosettaHoles and unsupported chemistry retain native definitions. No measured independent-versus-Rosetta speedup. |
| Consistent loss parameters and selectors | Shared chain groups, residue ordinals and DSL; no empty-to-all or duplicate-copy truncation; CMS reported side; strict normalization; startup and runtime use selector ownership. | Real selected-fragment/CMS/HB calculations plus controlled native seams. Full-structure BUNS environment and explicit fragment surface semantics are tested. |
| Reuse AF3-family preparation | Owner-scoped MSA rows, isolated CCD, Boltz asset loading, Protenix template parsing. | Exact native MSA encoder parity, cache/content/RNG boundary tests and actual AF3 coordinator lifecycle with optional dependencies doubled. Full native/GPU inference remains unqualified here. |

Candidate comparison selected Rust sc-rs for SC and the installable
`py-contact-ms 0.1.1` for CMS. The latter retains its surface construction;
pair tensors are processed in chunks and nearest buried-dot distances use a
KD-tree. Atom-neighbor construction still has a dense quadratic matrix.
`cms-cuda`'s CPU route did not supply an independent faster CPU implementation.
Holes candidates were rejected as substitutes because cavity measurements do
not reproduce the trained RosettaHoles score. See the
[loss definitions and sources](user_guide.md#61-结构-lossbackend选区与指标定义).

Cross-review found and fixed CMS A/B direction reversal, partial explicit-H
coverage, protonated carboxyl acceptors, empty surface queues, generic unknown
radii, hidden SC radius overrides, selector dependency omissions, stale alias
evidence after `_for` changes and cache hits deleting gate observations.
The earlier v2 snapshot layout was restored from an actual blob written by the
baseline code; absent backend evidence remains `None`. Pinned native Python
CCD code reproduced shared-base pollution before the fix and isolated custom
entries afterwards (C++ CIF/resource I/O doubled). All retained findings were
independently re-executed by the integrator or cross-reviewer.

The [recorded benchmark data](../examples/benchmarks/results/structural_losses_af3_preparation_20261116.json)
contains every timed round, dependency versions, input identity and parity.

| CPU component | Baseline median | New median | Correctness/work evidence |
|---|---:|---:|---|
| CMS, 1,412 heavy atoms, explicit side B | 13.8137 s | 4.2033 s | Five alternating rounds / ten successful calls; max total/per-atom error `1.07e-14`; 3.29× against public Python CMS, with visible shared-host timing variance. |
| Native AF3 MSA encoder, 2,048 × 512 | 245.23 ms | 5.16 ms | Five native/delta pairs; bitwise equality; exactly one newly encoded query row per mutant. Cold cache 263.22 ms. |

Reproduce CMS with `ASTROLABE_METRICS_BENCHMARK=1 python -m pytest -q -s
tests/test_independent_structure_metrics_benchmark.py`; the optional
`ASTROLABE_METRICS_BENCHMARK_OUTPUT` path retains JSON. Reproduce MSA with the
[existing profile command](adaptive_runtime.md#af3-family-input-preparation-reuse).
Neither measurement establishes complete featurisation, Rosetta, GPU or design
trajectory speedup. MSA-free inputs have much less reuse opportunity.

The first broad release CPU run completed with 1,758 passed, 81 failed,
17 skipped and 10 errors. Six failures from an untracked implementation file,
uninitialized release submodules and an uninstalled package were rerun after
tracking/installing the sources: all six passed. Remaining failure classes are
Unix-socket creation denied by this host, and the existing Mol-1 empty-options
`KeyError: repo_path`, reproduced unchanged on both baseline and implementation.
An unrestricted full pytest attempt stopped at 19 collection errors for absent
native AF3/AF2/PyRosetta/Boltz/RhoFold and related optional dependencies. These
runs are not reported as a green full-suite or native release qualification.

Final focused runs: 142 passed across the independent metrics, selector/backend,
evidence and AF3-family preparation suites; 166 passed / 21 native skips across
configuration, existing CMS/feedback and loss-expression consumers; 151 passed
across distribution, installation and container-alignment tests. A separate
Python 3.11 environment with the exact runtime geometry pins passed all 34 real
independent-library tests. The real-file LossEngine integration computes all
four independent metric families, retains backend records on a cache hit and
rejects any PyRosetta acquisition. Counts describe separate commands with
overlapping tests, not an accumulated unique total.

Host and container dependency manifests include the new packages; release CPU
CI includes the new regression suites. Paper, ligand benchmark and RFD4
protocols pin Rosetta to preserve historical definitions. Other migrated losses
prefer independent implementations by default. HB/BUNS methods and surface
radius tables differ, so existing thresholds require calibration or an explicit
Rosetta backend. The change is reviewable on a feature branch and does not
constitute a deployed native/GPU qualification.

The first remote CI attempt exposed an already-stale durable-recovery coverage
record: the unchanged baseline and this branch both rejected
`model_execution.py`. Review against the recorded `c8cdc4b5` anchor traced the
only covered-file change to the earlier `5400e797` Mol-1 `poolable` option.
The four journal/operation files and imported RNG helpers were unchanged;
the owner-selection change does not alter the modeled single-owner call
boundary. An independent review confirmed the correspondence. Updated the
coverage and project-skill review anchors to `21d6688b31a1170e0d197e650c533722ba31fcfc`,
without changing the model, runner, invariants or product code.

`python scripts/validation/verify_runtime_recovery.py --self-test --jar
<verified-tla2tools-1.7.4.jar> --output <fresh-directory>` passed with TLC 2.19,
OpenJDK 21.0.13 and Python 3.12.14. Both normal budgets passed (24 and 178
distinct states), the real-code suite passed all 18 cases before and after
mutation checks, and each of the three broken models and corresponding real
code mutations failed its required property. This reestablishes the existing
bounded recovery qualification; it does not add native model/GPU coverage.
