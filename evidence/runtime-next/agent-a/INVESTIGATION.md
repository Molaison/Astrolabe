# Agent A: initial read-only runtime investigation

Status: initial investigation DONE; implementation NOT STARTED. Source HEAD: `d66ee714f2e755948534f95d9d125287ce8dbd91`. Parent: `834402daade9f62e9dec0b08041dade163a89369`. Worktree: `/workspace/scratch/1a0b1639644d/cloud-20261104/agent-a`, branch `work/runtime-a-20261104`.

The existing untracked `resources/.paper_inputs_protein.tar.gz.zp8oMB` has been preserved. No production source has been changed or committed. The old Mental Model is at b537b8a; its guidance on repeat isolation, accepted-mutation versus stage-entry transactions, and preserving rejected-attempt history remains relevant, but current implementation claims below come from the pinned HEAD.

## Recommendation to lead: pilot R1

**The first performance pilot should eliminate redundant cold starts for a narrowly certified cheap CPU-only serial workload.** Historical W0 does no broker-dispatched model work: all 27 stage events in each resident run are Dummy CPU preparation/execution/finalization, with about 2–3 ms summed stage time per complete run. The additional ~23 s is outside these stages. A broad new scheduler or GPU batching scheme cannot target that measured cost.

A strict direct fast path is the smallest source change, but it has an important contract boundary: `execution.enabled=true` currently enters `run_parallel`, which constructs `ResourceLimits`, validates `ControlPolicy`, starts BrokerSession, and monitors aggregate RAM/output/process ownership. Calling the existing in-process `serial()` loop avoids these monitors. **Absence of an explicit `execution.resources` mapping does not prove absence of resource semantics: defaults are allocation-derived.** Therefore an unconditional `enabled && all_cpu` branch is not acceptable.

Two viable scopes for the pilot are:

1. Strict direct routing, only if the lead explicitly defines automatic execution as permitting this optimization for a certified built-in workload with no explicit resource/control/concurrency request. It must reject every explicit resource mapping (including seemingly harmless defaults), custom control policy, requested `max_active > 1`, model-backed component, custom selector/hook, and existing broker environment. All execution configuration must still be validated before routing. This can reuse the exact legacy repeat loop with its reset and seed semantics, and emit an explicit `serial_cpu_direct` route record. Remaining limitation: allocation-derived broker monitoring is bypassed and must be accurately documented, not silently claimed preserved.
2. **Preferred if current resource semantics must remain fully intact:** retain BrokerSession and run the certified cheap CPU repeats through one monitored child process using the existing serial repeat lifecycle. One process import is amortized over the already requested repeats; the broker still owns/monitors that child and output tree. Do not introduce threads around `_run_pipeline`. Reuse `run_repeats` for seed derivation and `global_config.reset`, and make parent configuration/seed ownership explicit to avoid deriving seeds twice. This should target ~6 s parent + ~8 s one child instead of ~6 s parent + 3×~8 s children in archived evidence. These are hypotheses, not new measurements. Explicit concurrency >1 retains the ordinary isolated path.

The lead should choose between those contracts before source editing. A route that saves ~23 s by dropping requested limits would be an invalid win. If a truly in-process direct route is chosen, the frozen harness should classify it as such instead of continuing to label its measurements as resident service.

## Caller-first architecture comparison (pstack architect)

| Design | Caller contract / data crossing boundary | Existing owners reused | Main benefit | Correctness cost / rejection criterion |
|---|---|---|---|---|
| A: certified cheap-CPU route, serial lifecycle | Caller supplies one unresolved Hydra config containing requested repeats; only runtime execution policy is selected; all scientific fields stay unchanged | `run_repeats.repeat_configs`, existing `serial`, `_run_pipeline`, `global_config.reset`, capability validation; monitored-child variant also reuses BrokerSession/run_trajectory ownership | Removes repeated launcher/import work for tiny CPU workloads | Must not silently discard resources, custom control, explicit concurrency, or GPU-capable plugins. Broad `CPU_EVALUATORS` is too permissive because it includes Rosetta. Initial certification should be Dummy + plain MCMC + sequence losses + built-in CPU initializer, without hooks/selectors. |
| B: preserve process topology, slim eager imports | Every trajectory remains a fresh process with its existing configuration DTO; defer expensive optional module imports to the actual caller that uses them | Current session, ledger, guardian, trajectory subprocesses; existing lazy app loader | Reduces launch overhead for all eligible trajectories while preserving current monitoring and isolation | Needs fresh import profiling: archived stage data only establishes cold-process overhead, not which import dominates. Moving imports can change registry initialization/optional-dependency failures; verify public factories and registry contracts. |

A persistent generic trajectory pool or pre-fork model pool is a larger, third option and should not be the pilot: it needs explicit reset of process-global registries, RNGs and framework state, and CUDA pre-fork behavior is not certified. Current `CPUExecutor` already supplies bounded pure-data subprocess work; it does not establish that the full mutable scientific pipeline is safe to pool.

## True control and data path at pinned HEAD

`Astrolabe.run()` -> cached `app._load_run_module()` -> heavy `run.py` import -> `run_repeats()` validates repeats and seeds -> unresolved per-repeat clones with deterministic seeds -> enabled branch `validate_parallel()` -> `resolve_active_limit()` + resident bounds -> `run_parallel()` -> ResourceLimits/ControlPolicy/AdaptiveController -> BrokerSession -> per-trajectory `run_trajectory()` -> fresh `python -m astrolabe.runtime.trajectory` -> `_load_run_module()` -> reset -> `_run_pipeline()`.

Model work, when a broker adapter exists, follows `BrokerClient` -> authenticated UNIX RPC -> `State.call` -> worker call lock -> pending-memory admission -> CPU/GPU compute admission -> `DaemonClient.call` -> resident worker -> resource observation -> release or retained staged payload. Scientific proposal generation, RNG, acceptance, and stage transitions remain owned by the trajectory process. `CPUExecutor` owns a separate bounded JSON task pool with ordered result delivery, pending/result byte caps and GPU-hidden subprocesses; it is not exercised by archived W0.

`astrolabe/__init__.py` and `app.py` are already light/lazy. Do not add another app loader or blame their import. `run.py` eagerly imports initializer families, structure helpers, DPLM wrapper helpers, losses and the scientific pipeline. Individual costly imports require fresh profiling before an import-slimming patch.

## Existing mechanisms to preserve

- Deterministic per-repeat seeds, unresolved Hydra cloning, raw-result repeat subdirectories, reset and graceful-stop loop already exist. Do not duplicate them in a new scheduler.
- ResourceLedger already bounds CPU slots, RAM, pending items/prepared bytes/output bytes and physical GPU compute; its `lease()` context already recomputes remaining timeout between queue stages.
- BrokerState staged prepare/execute/finalize retains the pending-resource reservation across phases and releases compute between phases.
- Physical GPU UUID/file locks and process-identity-aware guardians prevent claiming released VRAM while owned processes remain. Local restrictions must not be worked around by disabling those checks.
- AdaptiveController already separates cold/steady observations and requires enough evidence before growing. W0's cap-one behavior is not itself a demonstrated controller bug.
- Exact component allowlists and selector source identities exist. `CPU_EVALUATORS` includes `rosetta_crystal`; CPU safety does not imply cheap execution or absence of an opaque subprocess. A cheap-workload route needs a narrower rule.

## Executed probe and historical comparison

Executable probe: `evidence/runtime-next/agent-a/decompose_historical_trace.py`.
Raw derived result: `evidence/runtime-next/agent-a/historical_trace_decomposition.json`.
Reproduce from this worktree:

```bash
python evidence/runtime-next/agent-a/decompose_historical_trace.py > evidence/runtime-next/agent-a/historical_trace_decomposition.json
```

This reads archived evidence and is marked `derived_archived_evidence_not_new_benchmark`. It does not measure the current host. The archived protocol SHA is `f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5`, matching the lead's frozen W0: AAAA, Dummy evaluator, MCMC random mutation/position, sequence-residue-fraction objective, 3 repeats, seeds 17/18/19, 3 steps.

| Archived mode | Median complete run | Median pre-broker interval | Median individual trajectory process | Stage time summed over all 3 complete runs |
|---|---:|---:|---:|---:|
| direct | 9.6914 s | n/a | n/a | not traced |
| resident serial | 32.4176 s | 6.3350 s | 8.1426 s | 0.006963 s |
| resident adaptive | 33.4545 s | 6.5148 s | 8.4273 s | 0.008232 s |

The archived resident runs contain no request_queued/compute_start/device_start model-worker dispatch events. Stage sums are not end-to-end measurements; per-process durations include interpreter startup, imports and teardown. The trace supports locating the regression outside model admission/forward; an attribution to one specific import remains a hypothesis.

The original report records exact sequence/score/acceptance parity and release confirmation for complete attempt02 (27/27 trajectories). Attempt01's failed resident run is a retained negative result caused by process-observation accounting; do not pool its timing with successful throughput. The archived measurements used three rotated mode repetitions on f102; they are not same-host measurements against this task's eventual HEAD.

## Minimal falsifiable benchmark and independent validation

Reuse `scripts/validation/runtime_workflow_benchmark.py`, not an artificial speed loop. The lead froze five mode groups (15 trials), 900 s total CPU wall budget, the exact W0 protocol and reference source. Run in GitHub Actions where UNIX sockets and matching process namespaces are available, on one allocation with explicit thread/affinity controls. Keep baseline and candidate in isolated directories; direct baseline is pinned separately. Do not use a failure to launch the broker as a fast-path selection criterion.

For each successful run preserve exact command/config/protocol digest, elapsed time, process-tree RAM/CPU/output telemetry, source commit, release checks, score files and sequence/acceptance trace. Report useful completed trajectories divided by actual wall time, success counts and per-run distribution. Pilot keep requires unchanged requested repeats/steps/seeds/evaluation counts and exact W0 scientific parity, plus repeatable throughput improvement relative to the frozen execution-enabled baseline on the same runner. GPU claims remain excluded.

Independent tests: route eligible W0 to the intended path; explicit resource dictionaries/custom controller/explicit concurrency/model evaluator or mutator/hooks/plugins must retain ordinary behavior or reject correctly; malformed runtime config must still fail; stage transitions/stop/reset/repeat paths remain owned by existing lifecycle. Agent B can independently inspect backend/routing exclusions; Agent C can reproduce full per-repeat scientific parity including rejection and stage-entry cases.

## Subsequent independently owned A module

`State.call` end-to-end admission deadlines and boundary behavior are a candidate correctness module after the pilot. At current source, `remaining` is computed before `admit_pending()` and reused unchanged for `acquire_compute()`, unlike the existing ResourceLedger.lease context. A second wait can therefore receive an outdated budget. The dispatch then floors the remaining timeout to 1 ms. This is currently a **source-derived candidate, not an established bug**; reproduce with the real State/ResourceLedger and controlled blocking clients, then implement only if an expired request can cross the dispatch boundary. Tests should cover pending+compute sequential waits, exact identity/no replay, and reservation cleanup. This change is separable from B model lifecycle and C producer science.

Monitoring repeats process-tree and artifact-tree scans. These may matter for larger outputs, but no current run establishes them as a dominant bottleneck; do not add cache layers before measurement.

## Blockers and scope of evidence

Local AF_UNIX binding returns EPERM and process namespace verification is false; full broker/guardian runtime cannot be verified here. Do not disable these protections. Local direct W0 succeeds according to the lead's independently recorded baseline log. CPU dependencies are available through the shared venv. Real CPU resource validation/benchmarks are scheduled for authorized GitHub Actions, not yet evidence in this report. No allocated GPU, pretrained checkpoint access, /xcf26 mount, or SSH route exists in this child environment; GPU backend certification and destination transfer remain BLOCKED pending an authorized environment. No CPU/control-plane probe may be labeled GPU acceleration.
