# Recovered Runtime integration candidate

## Status and provenance

**DONE locally:** a new isolated worktree based on validated `9d62b3eb86cb75a2266af49dc48cb851e69e0341` reuses the recovered broader Runtime. The original remote fix/opt trees, earlier local snapshots, the main PR worktree, Agent B output boundaries, Agent C scheduler, R1 SciPy imports, and frozen measurement code are unchanged.

- `07d121901dc50eeb1c3a4a13d872648618b8978a`: eight Runtime owner files copied byte-for-byte from the preserved old fix, its related tests, explicit contract documentation, and adapted current deadline guards.
- `60684bd4fefa6e9b7282176e8b5af7700ef98093`: a separately reproduced admission-counter repair, plus a two-boundary regression. This is the final production source for this candidate; later evidence commits do not change it.
- Old source base: `d66ee714f2e755948534f95d9d125287ce8dbd91`; old branch `fix/runtime-admission-isolation-20261103`; full patch SHA-256 `49c84419808453d6629cf3f3464305590ea57f8a38e7b9655358ca61d1904f56`. Per-file source SHA values and deliberately deferred files are in `source-provenance.json`.

**PARTIAL:** real socket/process/isolation/saturation and the unchanged complete Runtime regression entry point still require root's capable-host run. These local tests do not certify a trained GPU model, new throughput, or the entire broader control plane. No scientific evaluations or GPU executions were run by Agent A in this integration step.

## Contracts being integrated

| Boundary | Resulting behavior |
| --- | --- |
| RPC → State → ledger → daemon | The caller's absolute monotonic deadline is propagated through socket framing, timed State/call admission, pending/compute admission and daemon transport. Earlier caller deadlines prevail. Cleanup and final ownership bookkeeping may still delay return after expiry; there is no hard wall-return promise. |
| queued / active request | One request owns its cancellation token and lease gate. A waiter cannot clear another dispatch's delegated state. Cancellation uses a separate control lane; its acknowledgement is not proof of release. |
| ingress | Existing business/control/nested lanes are reused. Business/nested capacities follow pending_items, control capacity is eight. Accepted partial frames count against capacity, reads are bounded, and a managed parent admits at most one nested RPC. |
| explicit rejection | `AdmissionRejected` states `not_executed` and retryable. The caller may make a new explicit decision; the Runtime never automatically retries scientific work. A lost response after submission stays `OutcomeUnknown`. Transport ID reuse after proven no-dispatch rejection does not weaken the independent candidate/scientific identity guards. |
| CPU and GPU uncertainty | Unknown CPU work fences the affected worker; independent CPU workers remain usable. Unknown GPU work fences its physical device. Unconfirmed cleanup retains reservations/artifacts. |
| staged ownership | A cancelled or expired execute/finalize that was never submitted retains the preceding successful prepare/execute reservation and scientific owner. An explicit subsequent action may resume or release it. No implicit replay or extra evaluation is introduced. |

The existing old-fix implementation supplies these behaviors; they were not rebuilt as a new competing controller. The previous narrow deadline/delegation fixes are subsumed by the recovered implementation and their seven shared outcomes remain covered. Current tests were adapted only where they prescribed private token spelling/signature/fake-clock behavior, or where the two explicitly approved contracts changed (CPU worker fencing and unsubmitted prepared retention).

## Verified correction to recovered code

Root found and independently reproduced a real regression in the recovered implementation. After successful compute admission, a scheduling delay can exhaust the request deadline before the timed State lock retires its waiting count. The old code used `admitted is None` in cleanup; `admitted` was already set, so `metrics.waiting` remained 1 despite zero busy requests, no dispatch, and empty reservations.

Agent A reran root's exact probe against committed `07d1219`: timeout after about 0.20 s, `waiting=1`, `busy=0`, `dispatches=0`, ledger released. The new test reproduces that boundary and also delays after the waiting count has already been retired. The red run is **1 failed / 1 passed**. The three-line production change records `waiting_counted` and clears it in the same State-lock critical section as the actual waiting decrement; cleanup retires the count only while that flag is still true. This distinguishes ownership transitions and prevents both the leak and double decrement.

On `60684bd`, the same independent probe reports `waiting=0`, `busy=0`, `dispatches=0` and empty reservations. The full deadline suite, including both transition cases, passes. Root separately reran the original probe on this exact fixed commit and confirmed the same result. `probe_waiting_cleanup.py` is the exact root probe; raw before/after logs and source digests are preserved.

## Local validation

| Command scope | Result |
| --- | --- |
| First current deadline suite on the unadapted recovered code | 2 passed / 8 failed; retained in `initial-negative.tar.xz`. Six failures were fixture coupling, two were the explicit contract differences described above; they were not mislabeled as eight production defects. |
| Adapted existing deadline suite on `07d1219` | 10 passed / 0 skipped. |
| New waiting-transition test before the repair | 1 failed / 1 passed / 10 deselected. |
| Complete deadline suite after the repair | 12 passed / 0 skipped. |
| runtime_context, runtime_controller, execution_auto_limits, runtime_contracts and runtime_resources, excluding five known process-observation cases | 31 passed / 0 skipped / 5 explicitly deselected. |
| Three non-socket ingress/client/signal-anchor cases | 3 passed / 0 skipped / 10 socket/process cases explicitly deselected. |

The final local gate therefore has **46 passing cases, 0 skipped, and 15 explicit environment-limited deselections**. Those deselections are not counted as certified process or socket behavior. Real State/ledger/locks and bounded CPU/file operations were used; controlled worker and post-response observation boundaries are explicitly documented in the tests.

C's independent three-case scientific probe on `60684bd` additionally confirmed: unknown work is dispatched once and not replayed while another CPU worker remains usable; queued cancellation is identity-scoped and does not imply release; staged cancellation preserves the original prepared owner and explicit resume/finalize produces exactly the requested calls before clearing reservations. It used controlled worker/observation boundaries, not actual IPC or GPU. The independent raw and probe are retained in the verification archive.

## Capable-host command

Run the checked-in script from the isolated candidate using an existing reviewed Python environment and a **new** absolute output directory:

```bash
ASTROLABE_TEST_PYTHON=/absolute/path/to/reviewed/python \
ASTROLABE_TEST_OUT=/absolute/path/to/new/runtime-reuse-validation \
bash evidence/runtime-next/f102-runtime-reuse/capable-host-validation.sh
```

The script requires Linux process-namespace and UNIX-socket preflight through the existing `benchmarks/runtime/validate.py`, requires at least two effective CPU slots, records HEAD and pip freeze, and runs:

```bash
python benchmarks/runtime/validate.py regressions --out "$OUT/regressions"
python -m pytest tests/test_runtime_ingress.py \
  tests/test_runtime_request_isolation.py tests/test_runtime_saturation.py \
  -q -ra --junitxml="$OUT/recovered-contracts.xml" --basetemp="$OUT/pytest-tmp"
```

The existing complete regression gate remains unchanged. The recovered suites use `ASTROLABE_EXPECT_GUARDRAILS=1`; JUnit must contain at least one case and **zero failures/errors/skips**. No `-k` deselection is used on the host. GPU visibility is cleared, CPU library threads are bounded, and per-command timeouts are 600 s for the complete existing gate and 300 s for recovered contracts. Root owns any lower allocation/overall experiment budget. Raw stdout, JUnit and per-scenario files are preserved. The shell script was syntax-checked locally; it was not claimed to have run here.

## Preserved workspace incident

The initial uncommitted `copy2` transfer was observed to revert nine tracked files to exact validated-base bytes after the first negative tests; new files remained. Root and A confirmed separate worktree/index ownership and no other agent writes. After another `copy2`, old-mtime Runtime sources reverted while the staged index and fresh-write docs/test remained. The cause is **UNKNOWN**: a one-off disposable git repository with an expected failing pytest did not reproduce a rollback. No unsupported causal claim is made.

All original diffs, untracked bytes, SHA/mtime observations, commands and the disposable negative probe are retained under `incident/` in the verification archive. The candidate was restored from its verified staged contents using fresh content writes and immediately committed before validation. Its source hashes remained stable through the subsequent tests. Original worktrees were not restored, reset or overwritten.

## Deferred work and performance interpretation

The old workflow-report completeness gate and `performance_cohort` report correction remain preserved in the old source patch and are deliberately deferred: the current frozen comparison wrapper independently enforces complete workload/scientific gates. The eight Runtime owners are the only changed production files. Old PyRosetta lazy loading/source scanning, B's ESMFold helper, C's scheduler and the validated R1 SciPy imports are outside this integration.

No speedup is inferred from these tests. Any performance decision for this broader Runtime must compare the candidate against both validated `9d62b3e` and d66 with root's frozen workloads, actual completed evaluation counts, scientific comparison and explicit budget. The prior measured R1 result is not silently transferred to this new candidate.
