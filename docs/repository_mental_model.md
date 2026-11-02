# Repository Mental Model

Baseline: `xcflab/Astrolabe@b537b8a81ff0738922e80a997ffeb9700cdafe09`. This describes the architecture-convergence implementation built on that baseline, not an already merged or deployed release. Check the current commit before reuse. See [implementation and validation limits](architecture_convergence.md).

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
- Explicit repeat seed increments by repeat index; missing seed derives deterministically from task/index; zero is invalid. Bare shared chain selectors are chain-first; `protomer:A` expands repeated A copies.
- `run --dry-run` stops before model execution; `--print-config` alone does not. Experimental candidate-batch dry-run still writes candidate configurations.

## Verification routing

Use affected-owner tests and real CPU boundary probes before broad checks. Decision changes require accepted/rejected history, CSV rows and RNG replay; evidence changes require clone/filter/merge and transformation provenance; resource changes require acquisition/cleanup faults and bounded scheduling. Native model, GPU, end-to-end and measured throughput claims need separate evidence and were not established in this change.
