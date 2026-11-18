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

## Configuration and mutation boundaries

This work starts from `09f0460feb733b4675923ee580c977c7f7423e13`. The requested
outcome is a usable task configuration surface, clear parameter ownership,
effective agent conventions and one mutation-policy interface. The
[ownership contract](repository_mental_model.md#configuration-and-mutation-ownership)
records the implemented boundary. The reviewed code snapshot is
`686e3fb53a13f877001b7be3c8e611b14e40d34f`; the later ledger update changes
documentation only.

### Baseline evidence

| Observation | Evidence | Meaning |
|---|---|---|
| Architecture/training overrides pass backend preflight. | Executed `validate_backend_settings('evaluator', 'protenix', ..., check_paths=False)` for `c_s=1` plus `adam.lr=42`, `model.pairformer.n_blocks=1`, and `data.train_sets=['invented_train_set']`; all returned no errors. | The public boundary accepts values that are outside the intended inference interface. This does not prove that native inference uses each value. |
| The adapter already restores much of the model preset. | `evaluators/wrappers/protenix.py::_normalize_protenix_configs` builds upstream defaults and model-specific settings, then merges selected overrides. | Many exposed settings are silently ignored; `data` and `model.constraint_embedder` still receive whole nested mappings. Removing misleading fields and rejecting overrides are separate requirements. |
| Component interpolation becomes an initialization snapshot. | Executed a merge with `count: ${budget}` and budget 10, then changed the source budget to 3. The source resolved to 3 and the selected component remained 10. | Live mutation state needs proposal-time resolution. An initialization interpolation cannot supply that contract. |
| Published environment defaults contain a personal installation. | `config/env.yaml` contains `/xcf26/zpzeng/scientific-tool-runtimes/...` for Mol-1 paths. | Machine paths belong in local environment choices, with portable defaults or explicit missing assets. |
| Checkpoint ownership already has a local precedent. | `mutators/wrappers/ligandmpnn_run.py` reads `num_edges` and `atom_context_num` from the checkpoint and constructs the fixed architecture internally. | Reuse backend ownership rather than inventing a universal schema for every vendor's internal config. |

The two executed probes used Python in the task's isolated CPU environment.
They did not initialize a model, allocate a GPU or invoke a native backend.

### Design decisions

Retain Hydra composition, family registries and the existing startup boundary.
Keep the task base small, move backend defaults to their owner, and keep
environment-specific values in environment configuration. Internal defaults are
not public overrides merely because they are represented in YAML. The selected
backend validates the supported surface before model construction.

A full migration to Structured Config classes was considered. It can improve
type checks, but would add a parallel schema across many dynamic aliases and
vendor interfaces before fixing ownership. Prefer a bounded public contract
now. Splitting the large YAML without rejecting unsupported fields was also
rejected because it only relocates the existing ambiguity.

Mutation policy keeps one stage input and one typed resolver. Resolve the
policy for each proposal using current eligible positions and the selected cap.
Treat negative sentinel values, booleans and nonintegral counts as explicit
migration errors. Preserve native joint/reconstruction semantics and existing
scientific constraints rather than truncating a coupled output after sampling.

Common public names require actual semantic equivalence and a conflict rule.
Similar backend names alone do not establish equivalence. The uploaded RFD4 /
Mol-1 profile contains deliberate missing inputs and unconnected scores; its
migration must preserve them as missing values.

### Acceptance status

The implementation is integrated. `base.yaml` is 61 lines; base plus the four
component files is 640 lines, down from the original 1,075-line base. Training
and model-architecture copies were removed, while supported inference controls
remain with their adapters. The public names `recycles`, `samples` and
`diffusion_steps` normalize per layer; conflicting spellings fail. All nine
evaluator families retain their configured inference defaults against the
pinned baseline. Portable `env` is now the default; existing local installations
select `env=env_local` explicitly. RFD4 paths are present in both portable and
container environments.

Mutation policy is owned by `mutation_budget.py`, with stage input resolved
against current shared positions for each proposal. Tests cover shared-site
deduplication, locked/PTM exclusion, same-stage length changes, zero and empty
eligibility, stage switching, weighting and no added RNG consumption. MCMC's
per-alias `max_mutations` remains a distinct local ceiling required by existing
paper protocols; it uses the same resolver. Proton's native cap is derived per
proposal. Equal-length RFD4 budgets released positions; explicitly enabled
variable reconstruction remains a whole joint proposal governed by its own
length/structure constraints, not a point-edit cap. Portable native/beam
profiles still require integer budgets.

| Executed check | Result and scope |
|---|---|
| Baseline startup/backend/merge/stage/factory/RFD4 CPU set | 194 passed at `09f0460`; no model execution. |
| Final affected CPU regression set | 1,358 passed, 3 failed, 2 deselected at `686e3fb` (84.45 s). Failures and deselections are specified below; this is not an all-green full-repository claim. |
| Independent fixed-snapshot review | 163 passed across public contracts, mutation budgets, CLI command families/repeat symmetry and environment defaults; no retained reproducible finding. |
| Native shell startup boundary | 40/40 corpus cases passed: 32 invalid cases and 8 valid controls. The sensitivity control confirmed that the observer detects entry into scientific runtime. |
| Real CPU CLI trajectory | The sequence evaluator and MCMC completed root plus two proposals: 20 eligible sites with `25%` produced 5 changed positions in each proposal. No model weights or GPU were used. |
| RFD4 factory and uploaded-profile probe | Actual registry factory received all four environment paths. All 12 original `???` paths, effective generation controls and scoring intent were preserved. The profile remains incomplete. |
| Protenix request ownership | Direct native ConfigDict and serialized daemon paths passed CPU fake-native probes. Large→medium→small requests derive independent precision flags; external architecture overrides still fail before native imports. The integrator independently reran both reviewer probes. |
| Configuration and installation | 78 paper profiles composed in the configuration worktree. Final wheel-content and extracted-install tests passed with all four component files. |

The final affected set covered `test_public_component_contracts`,
`test_mutation_budget`, backend/task formal corpus, startup/backend validation,
merge diagnostics, CLI dry-run/command/repeat suites, environment wiring,
selected evaluator/mutator factory suites, the six changed validation-harness
suites, wheel resources, RFD4, ESM3, Proton, MCMC/PoreBlocker, mutator runtime and
pre-update behavior, stage defaults, complex-pose consumers and Protenix wrapper
CPU tests. The two `normalize_protenix_v2` tests were explicitly deselected
because the native Protenix runtime is unavailable.

The three failures were independently reproduced unchanged at `09f0460`:
`test_ligand_preparation_is_cpu_and_does_not_consume_rng` lacks `prody`,
`test_carbondesign_actual_stages_keep_probabilities_and_mask_slicing` lacks
`ml_collections`, and
`test_cpu_model_factory_with_real_broker_releases_failed_request` cannot create
an AF_UNIX socket in this execution environment. No production check was
weakened to hide these limits. A separate pre-existing typed-recovery fixture
used an undeclared weight-free backend name; its test now uses the existing
`fake` contract so the crash/resume test can exercise recovery while preserving
production asset verification.

Reproduce the startup boundary with:

```bash
python tests/run_formal_config_cli.py --filter startup --jobs 4 \
  --report /tmp/astrolabe-config-startup.json
```

Reproduce the real CPU trajectory in a configured CPU environment with:

```bash
ASTROLABE_RUNTIME_ACTIVE=1 GH_CUDA_PREINITIALIZE=0 CUDA_VISIBLE_DEVICES='' \
python -m astrolabe.cli run \
  --standalone tests/fixtures/formal_config/startup/valid_minimal.yaml --no-chdir \
  env=env out_dir=/tmp/astrolabe-mutation-smoke task_name=mutation_budget_smoke \
  steps=3 'stages.default.mutation_num=25%' ananas_path=null
```

The five-agent workflow separated configuration analysis, configuration
implementation, mutation policy, prior-art/governance and independent review.
Implementation worktrees were isolated; the integrator owned shared CLI,
consumer migrations and final evidence. No native numerical-equivalence,
real-model end-to-end or GPU-performance claim follows from these CPU checks.

### Merge review follow-up

The independent merge review reproduced a startup regression in
`container_protenix_smoke` and `container_ligandmpnn_smoke`: their retained
`diffusion_batch_size` and `diffusion_chunk_size` fields were not consumed by
the old wrapper and were rejected by the new public contract. Both fields
were removed from these profiles and the Protenix branch of
`container_ligand_matrix_smoke`. The configured recycle, sample and diffusion
budgets are unchanged. All three actual CLI dry-runs passed with Protenix
selected, and a control confirmed the unsupported fields still fail validation.
The reproduced issue now has three CLI regression cases.

Two existing test consumers now read `samples`, retaining the value of five.
AF3 lifecycle checks use the existing cold asset fixture and local structure/MSA
inputs instead of `/case`. The registry construction check uses composed
defaults and path-only cold assets; Mol-1's missing source is explicitly rejected
without bypassing its pinned-source or checkpoint contract. Other registered
families retain their descriptor and unloaded-owner assertions.

The recorded recovery dependency diff from
`c8cdc4b5f7e393a26bd0b1fe7fb00e92c9759d99` to
`eec8ee27caccdcfc109b15ee19bf3cacb06a5901` changes only
`NativeModelClient.poolable`, which selects private versus pooled model ownership.
The four journal dependency files and `Runtime.call` transaction ordering are
unchanged. After checking that correspondence, the coverage hash and skill
review anchor were advanced. The full recovery self-test passed: both normal
models, all three named counterexamples, the real crash tests and their three
broken variants, and the restored source. Each real-code invocation ran all
18 cases without skips. The bounded model and its invariants were unchanged.

Review evidence uses the licensed a13 SIF's Python 3.11.14 with the existing
read-only startup/test overlays, a private zsh runtime and mounted OpenJDK 8u504;
no model or GPU was invoked. The separate migration, registry and environment
control suites passed 53, 20 and 83 cases respectively; these sets are not added
together. An eight-case `09f0460` comparison reproduced the Mol-1 missing-config
failure and four `/case` failures on the baseline. Full repository regression
and GitHub release checks remain required before merge.

### Primary-source checks

The repository pins Hydra 1.3 and OmegaConf 2.3. Research first established
composition and schema behavior, then checked the versioned OmegaConf docs and
native Protenix construction to distinguish documented behavior from inference.

- [Hydra Defaults List](https://hydra.cc/docs/advanced/defaults_list/) documents `_self_` and composition order.
- [Hydra Structured Config schema](https://hydra.cc/docs/tutorials/structured_config/schema/) describes validation of YAML against a schema.
- [OmegaConf 2.3 usage](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html) distinguishes lazy interpolation from resolved conversion and explains read-only/struct flags.
- [OmegaConf 2.3 resolvers](https://omegaconf.readthedocs.io/en/2.3_branch/custom_resolvers.html) documents caching by resolver argument strings, which does not supply a live runtime dependency.
- [Protenix model presets](https://github.com/bytedance/Protenix/blob/main/configs/configs_model_type.py) and [inference construction](https://github.com/bytedance/Protenix/blob/main/runner/inference.py) establish the model-name/preset boundary. [Inference controls](https://github.com/bytedance/Protenix/blob/main/configs/configs_inference.py) are defined separately.
- [Pinned Protenix model implementation](https://github.com/bytedance/Protenix/blob/2475421477ab414b571149ad4a875c390ff8a35d/protenix/model/protenix.py) confirms noise/guidance controls are consumed by inference, while mini-rollout counts belong to training; the latter and request-derived precision are not public task overrides.

## Native vdM and normalized pH metric integration

The qualified vdM pair proposal, role projection and local Proton adapter are
integrated through the existing native mutator factory rather than restoring
the historical base/runtime wholesale. Fixed/target constraints, starting-root
site identity, proposal budgets and current ResourceOwner/seed boundaries remain
owned by their existing components. Frozen historical runs are not migrated.

The follow-up repair on `04cd241` aligns the startup and factory contracts for
`vdm_pair` and `vdm_proton`, partitions the live stage mutation budget across
aliases, and passes the source-structure sequences through the Proton JSONL
runner. Known vdM infeasibility uses the existing refinement rejection type
and is also recorded by the serial optimization loop. Cached pairsets use
manifest-declared parent IDs and still recognize the flat single-parent layout.

The CPU CLI controls exercised all three daemon-option validators. The budget
control produced two edits from the first alias and one from the second under
a shared budget of three. The empty vdM library control used an existing
structure and recorded two no-match attempts after one initial evaluation.
Both attempts retained the accepted sequence and left the candidate score
blank. Native refinement completed two rejected proposals at the proposal
cap with only its initial evaluation. A zero-budget vdM control recorded
the corresponding budget rejection.

The configured Score worker used the real Proton checkpoint on CPU with
one broker CPU slot. It completed one state-scoring request and one
evaluation. A separate CPU pre-update control changed one position before
Proton. The Proton response recorded different source and working sequences
for the binder, matching source and working sequences for the context
chains, and eight engine candidates. The control used a root-only structure
evaluator, which stopped at the following evaluation.

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

### PR #16 integration with current main

The merge against `38bf2d4` retains root prediction receipts, profiling
validation and vdM component defaults alongside the new configuration layout.
Native asset verification now follows the SurfDesign and SurfPro checkpoint
locations used by their runners. The registry check uses an empty working
directory and model-free fixture files at those locations.

The registry, AF3 lifecycle, startup validation and Proton mutator selection
passed 157 local CPU cases. The recovery coverage manifest matches the seven
affected-source hashes from main. The earlier unrestricted test run encountered
storage errors and missing external fixtures. Release verification uses the
repository's clean-install CI matrix on the merged source.

### Task parameter defaults

Tasks composed from `base.yaml` inherit `steps: 20`, `repeats: 1` and
`seed: null`. The repeat planner derives a stable seed from the task name and
repeat index when the seed is omitted. Task and CLI overrides take precedence.
The RFD4/Mol-1 partial template inherits these three fields and keeps its
required case inputs and scoring placeholders.

The public CPU CLI with the dummy evaluator recorded steps 0–19 in one run
with a derived seed. Explicit overrides recorded steps 0–2 in two runs with
seeds 42 and 43. The saved configs and provenance matched both runs. A dry run
of the incomplete RFD4/Mol-1 template still rejected missing case inputs.
The configuration routing and CLI dry-run suites passed all 31 cases.

### Loss, evaluator, and mutator review

The local review started at `04cd241`. A retained RFD4 structure crashed
PyRosetta when the B-chain slice lost the connection between residues 69 and
101. The same input produced a finite `ph_core_network` value of `0.0` after
the slice preserved the bond. Preflight and `LossEngine` now accept the same
oligomer selector forms and identify invalid fields by their configuration
paths. The holdout `A=0.25`, `AA=0.75` selected `0.75` for `AA` in all three
tested selector forms.

The Protenix output check used two step directories with the same native CIF
filename. A missing current-step CIF now produces an error with the step,
oligomer, seed, and sample. The AF3 output check recovered two pLDDT values,
50 and 60, where the baseline merged residues numbered `1` across chains
and returned 55. These evaluator checks ran against local output fixtures.

A local `steps=20` run used the sequence evaluator and two equal-weight MCMC
aliases with budget 1. It saved 20 score rows, including initialization. Each
of its 19 proposal rounds changed one position in the first alias and zero
in the second. A separate XMix CLI run specified budget 9 over two independent
segments. A and B share the first segment, and C owns the second. The same
input changed 16 unique sites before the fix and nine after it. The Protenix
CUDA-only check was skipped in the CPU environment.
