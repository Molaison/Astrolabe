# Astrolabe working conventions

Start with `docs/repository_mental_model.md`, compare the current commit to its baseline, and investigate only changed owners and consumers. A proposal, a test file, an executed test, a native-backend run, and a deployed capability are different evidence levels.

For configuration changes, read the mental model's **Configuration and mutation ownership** section. Assign each field to task intent, environment, backend invariant or current runtime state before exposing it. Trace composition, validation, factory, consumer and saved-config behavior together. Keep one public name for one concept; reject conflicting aliases, unsupported overrides and missing required values before model initialization. Model presets and checkpoint metadata own architecture and training settings.

Keep mutation policy in `stages.*.mutation_num` and resolve it through the shared policy owner for every proposal against current eligible positions. Preserve its distinction from native sample count and realized sequence differences. Native joint or reconstruction proposals must declare their budget semantics; trimming a coupled sample afterwards is not budget enforcement. A changed boundary needs a valid control and a failing configuration or changing-state reproduction, with executed results recorded in the implementation ledger.

Preserve shared Segment identity, fixed/locked/PTM/covalent constraints, mutator and pre-update order, RNG consumption, last-accepted comparisons, gates, prefix stopping and repeated-step assessment rows. Mutation acceptance and stage entry have separate commit boundaries; rejection still consumes randomness and creates history.

`RuntimeState` owns registered storage; the legacy facade forwards to the active store and explicit `RuntimeContext` instances stay bound to their store. Legacy global consumers are still process-scoped. Do not infer concurrent-trajectory safety from the new context type.

Use `StandardScores.score_field_names()` for numerical fields and the evidence helpers for metadata. Do not invent sample correspondence from array positions or equal sequence strings. Original prediction evidence and derived structures are distinct.

Reuse family registries and explicit resource ownership. Parallel implementation uses isolated worktrees and one integrator for shared orchestration. Scheduling must not silently replace K=1 with best-of-N or first-completed search.

Validate meaningful behavior, compare with a pinned baseline where possible, and independently rerun retained reviewer repros. Separate CPU/control-flow, real-backend, end-to-end and performance evidence; obey task-specific testing limits. An audit is not a checkpoint and bounded CPU work is not evidence of GPU speedup.

Update the mental model and implementation ledger incrementally rather than creating competing architecture documents.
