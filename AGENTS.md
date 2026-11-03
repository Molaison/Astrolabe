# Astrolabe working conventions

Start with `docs/repository_mental_model.md`, compare the current commit to its baseline, and investigate only changed owners and consumers. A proposal, a test file, an executed test, a native-backend run, and a deployed capability are different evidence levels.

Preserve shared Segment identity, fixed/locked/PTM/covalent constraints, mutator and pre-update order, RNG consumption, last-accepted comparisons, gates, prefix stopping and repeated-step assessment rows. Mutation acceptance and stage entry have separate commit boundaries; rejection still consumes randomness and creates history.

`RuntimeState` owns registered storage; the legacy facade forwards to the active store and explicit `RuntimeContext` instances stay bound to their store. Legacy global consumers are still process-scoped. Do not infer concurrent-trajectory safety from the new context type.

Use `StandardScores.score_field_names()` for numerical fields and the evidence helpers for metadata. Do not invent sample correspondence from array positions or equal sequence strings. Original prediction evidence and derived structures are distinct.

Reuse family registries and explicit resource ownership. Parallel implementation uses isolated worktrees and one integrator for shared orchestration. Scheduling must not silently replace K=1 with best-of-N or first-completed search.

Validate meaningful behavior, compare with a pinned baseline where possible, and independently rerun retained reviewer repros. Separate CPU/control-flow, real-backend, end-to-end and performance evidence; obey task-specific testing limits. An audit is not a checkpoint and bounded CPU work is not evidence of GPU speedup.

Update the mental model and implementation ledger incrementally rather than creating competing architecture documents.
