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
