# Repository Mental Model

The architecture-convergence work was based on
`b537b8a81ff0738922e80a997ffeb9700cdafe09` and merged in `2a8b72a`.
The runtime integration combines those owners with the locally qualified
`25a2b3a` source snapshot, using their original `261b240` release tree for the
three-way comparison. Both histories are retained. Check the current commit
and [runtime qualification](adaptive_runtime.md#qualified-release-boundary)
before reusing evidence. The older [implementation ledger](architecture_convergence.md)
records the scope of the convergence work.

`configuration.py` also routes explicit typed search to `design/`, whose durable
raw/Score receipts and scientific identity checks own recovery. The observational
`execution_audit.json` described below remains a separate artifact; it cannot
authorize replay of an uncertain inference request.

```text
CLI / Hydra -> Astrolabe.run -> run_repeats -> _run_pipeline
  -> RuntimeState + shared Protomer / Segment graph
  -> initializer -> mutators / LossEngine / StageManager / evaluators
  -> isolated outgoing Input -> evaluator -> StandardScores + invocation evidence
  -> derived structure hooks -> DecisionScoreContext -> Decisions
  -> separate StageTransaction / gate / reassessment
  -> compatible scores.csv + provenance.json + observational execution_audit.json
```

| Change | Owner | Main invariant / validation |
|---|---|---|
| Runtime data | `core/global_config.py` | One store, facade forwarding, contexts remain bound across reset. Legacy callers still require process isolation. |
| Shared design / edits | `core/protomer_config_parser.py` | Object identity, pending vs accepted, pre-update rollback and locked/fixed/PTM/covalent constraints. |
| Prediction / evidence | `core/folding_input.py`, `evaluators/{utils,evidence}.py` | Isolated outgoing transport graph; numerical and evidence fields merge separately; unknown sample alignment stays unknown. |
| Assessment / acceptance | `losses/base.py`, `decision_context.py`, `decision_maker.py` | Higher is better; last accepted baseline; hard gates; rejected evidence can affect prefix stopping. |
| Stage entry | `stage_manager.py`, `stage_transaction.py`, `stage_transition.py` | Committed source may differ from execution target. A rejected direct stage entry does not undo an already accepted mutation. |
| Structure transformation | `structure_processors/`, `post_eval_hooks/` | Changed geometry is not a new prediction. Keep original confidence and derived-path provenance distinct. |
| Resources / overlap | `runtime_support.py`, backend lifecycle methods, `run.py` | Explicit deduplicated cleanup; serial prediction, bounded ordered CPU finalization. |
| Persistence | `decision_maker.py`, `run_config_lifecycle.py`, `execution_audit.py`, `posthoc/` | Same CSV row/column semantics. Single-file atomic replacement is not exact resume. |

## Constraints that are easy to miss

- `RuntimeContext()` captures the active store; `reset()` replaces the facade's store without retargeting existing contexts. Use init/update/set methods, not registered public module assignments. This does not isolate native backend globals or make legacy concurrent runs safe.
- A frozen Input can contain mutable lists. Snapshotting at dispatch isolates caller inputs while preserving backend-expected types; it does not alter native inference or sample selection.
- StandardScores numerical/native values retain their existing shallow sharing semantics. Evidence has explicit clone/filter/merge rules. Equal metric names do not prove cross-backend scientific equivalence.
- Unaligned candidate groups remain evidence, not a new selection policy. Input seeds are observed values, not proof of backend RNG consumption or exact sample identity.
- Attempt identity, accepted revision and assessment identity differ. Same-step stage reassessment creates another observation without another accepted mutation. Rejected attempts and initial reference rejection must remain visible.
- Execution-stage observation does not enter backend runtime dictionaries. Native completion, observational failure and scientific acceptance are different statuses.
- Terminal audit is best-effort evidence, not a live checkpoint. Absence of a completed assessment does not prove no partial effect occurred; uncatchable termination may produce no current audit.
- Main-loop overlap evaluates serially and bounds outstanding CPU finalizations. Experimental candidate batch remains a separate subprocess-based search protocol, not unified GPU scheduling.
- `design.collection.CompletionPool` validates a finite generation before admission or dispatch. Job IDs remain unique across the pool lifetime, including failures and consumed results. A rehydrated job can be reconciled once using the owner's terminal admission proof, without execution; missing proof retains the unknown fence. Worker results must match the submitted job ID and generation and retain its input payload. These guards do not change collection selection, RNG or budgets.
- Explicit repeat seed increments by repeat index; missing seed derives deterministically from task/index; zero is invalid. Bare shared chain selectors are chain-first; `protomer:A` expands repeated A copies.
- `run --dry-run` stops before model execution; `--print-config` alone does not. Experimental candidate-batch dry-run still writes candidate configurations.

## Verification routing

Use affected-owner tests and real CPU boundary probes before broad checks. Decision changes require accepted/rejected history, CSV rows and RNG replay; evidence changes require clone/filter/merge and transformation provenance; resource changes require acquisition/cleanup faults and bounded scheduling. Native model, GPU, end-to-end and measured throughput claims need separate evidence and were not established in this change.

## P0 failure and cleanup protocols

- DPLM and RAPiDock freeze each request once. A lost, malformed, mismatched or
  unmaterializable response after send is `RequestOutcomeUnknown`; neither a
  daemon restart nor a cache hit may bypass that fence. DPLM does not draw a
  replacement seed. This adds no cross-process DPLM bank/cursor recovery.
- Socket inference likewise treats partial-send failure and unproven response
  identity/materialization as unknown. Explicit pre-execution server rejection
  remains a known failure. Collection propagates unknown exception chains to
  the durable owner instead of publishing a failed terminal job.
- `ResourceOwner` keeps per-resource `CleanupRecord` receipts and returns a
  `CloseReport`. Failed references stay reachable; ordinary repeated close
  does not retry them. Only an explicit `close(retry_failed=True)` does.
  A concurrent close reports draining, never successful release. Startup and
  run scopes preserve the original exception and attach cleanup diagnostics.
- Native refinement and NA-MPNN retain inner owners after cleanup failure.
  DPLM/RAPiDock clear process handles only after owned process-group exit.
  Cleanup is called outside lifecycle locks. Cleanup receipts prove adapter
  completion, not unmeasured GPU allocator release for every backend.
- The CLI retains raw artifacts if cleanup is unresolved. Durable prediction,
  Score, collection and control receipts remain separate; journal identities,
  RNG and charged attempts remain authoritative. Files, futures and logs
  cannot authorize replay. Arbitrary cross-version checkpoint migration and
  all-backend recovery are unsupported.

The release CPU suite includes protocol fault injection, raw-only Score
recovery and child-owner cleanup checks. Native scientific qualification and
deployment evidence are recorded for the exact release separately.

## Scientific restoration and initialization protocols

Runtime contexts retain an immutable run identity. Old explicit stores remain
readable; pipeline/legacy-global operations reject an inactive store. Legacy
snapshots remain live-object projections and now validate their run/topology and
materialize the entire restore before writing any scientific field. Failed
capture does not leave an active context behind. The existing typed decision
state retains attempt, accepted revision, assessments and rejection history;
mutation acceptance and stage entry still commit separately.

Configured refinement checkpoints DPLM prefetch, pore activation and RFD4
feedback state in its original Runtime at proposal boundaries, including
consumption by rejected proposals. Recovery validates prior proposal receipts
before initialization and restores all selected projections before new mutation.
Missing/stale state cannot authorize continuation; unknown stays unknown.
DPLM requires `response_cache=false` for this durable profile. Predocker bank
projections alone do not cover dynamic-site/sidecar state, so configured durable
refinement rejects predockers before initialization. The legacy trajectory,
pre-update stacks, dynamic stages and live poses retain their existing live-only
or explicitly qualified portable boundaries. No arbitrary checkpoint migration
is provided.

Initialization proof and current residency are separately observable.
Partially failed pipeline/model initialization cannot be retried by ordinary
access. A process-level PyRosetta owner verifies actual native readiness and
binds options, params contents, version and database location. Conflicting
explicit requirements fail; external initialization is unattested. Provider
activation/reset cannot erase applied patches or a poisoned process.
MSA entries publish only after complete loading. Score extraction binds working
sequences, dynamic sites and actual loss parameters to its run; its cache is not
a durable Score receipt. DEBUG writes require `DEBUG=1` and remain debug-only,
with no RNG/recovery guarantee.

Posthoc collection binds input contents and derivation configuration. Records,
per-derivation outcomes and run completeness commit in one DB transaction;
PARTIAL is retried and successful independent derivations are reused.
Filter/export/archive completion is defined by the final content manifest;
multi-file promotion can remain PARTIAL. Original errors survive cleanup/proof
write errors. Backend adapters retain failed cleanup handles; shared Proton
requires one resource scope. Vendor training recovery is unchanged; only
reachable Proton fork-pool and Squashfs lifecycle boundaries were repaired.

## Proton integration boundary

`proton_pottsmpnn` / `Proton-PottsMPNN` are lazy, protein-only joint sequence
mutators registered through the existing factory and merged component defaults.
The persistent external runner also supplies fixed-context pH state scores;
private and explicitly shared runners attach to the active `ResourceOwner`.
Standalone posthoc callers still close their own runners. Raw extended tokens,
centres and native energies remain mutation evidence, not prediction confidence.
`ph_sensitive` and `ph_core` register through the existing loss loader. Posthoc
`proton-states` and `proton-parity` are explicit opt-in commands, not automatic
scientific acceptance. Patched source and external checkpoint/EV6 assets have
separate provenance; see [installation](proton_installation.md) and
[pH interfaces](ph_sensitive_design.md). This port adds no AF3 or vdM implementation.
