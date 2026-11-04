# Agent A runtime correctness and measurement delivery

Source baseline: d66ee714f2e755948534f95d9d125287ce8dbd91. All production changes remain in the assigned worktree; the preexisting untracked resource archive is preserved.

| Commit | Scope | Evidence |
|---|---|---|
| ad1037d | Frozen CPU comparison wrapper and GitHub workflow | 25 harness tests; real 30-trial plan with fixed W0 hash; local environment preflight truthfully BLOCKED |
| f1bef6f | Initialize evidence path in a runner step | Actual GitHub rejection: job env cannot use runner.temp; explicit runtime step fixes context availability |
| 54031fe | One State.call budget across worker lock, pending memory, compute and dispatch | Real State/ledger probe: 1 s budget dispatched at 1.303 s before; TimeoutError at 1.001 s and zero dispatch after. Red 2 failed/4 passed; broader local gate 40 passed/6 namespace skips |
| 1055b86 | Scope PYTHONPATH to regression subprocesses | Actual CI 220 passed/4 fixture-import failures; local child entry reproduces ModuleNotFoundError before and hello/EOF success after; benchmark environment unchanged |
| 00a8b68 | Only call_lock owner clears delegated request | Independently rerun C's real admission probe; timed-out waiter cleared active delegation and caused TypeError before; first call succeeds after. Concurrent positive/negative gate red 1 fail/1 pass; 16 deadline/contracts tests pass |
| eac3081 | Include entry bookkeeping in admission elapsed time | 30 ms budget plus 120 ms entry lock dispatched with fresh budget before; after releases the lock then rejects without dispatch. Red 2 failures; 18 deadline/contracts tests pass |

The deadline probes use actual State and ResourceLedger admission and actual locks/condition waits, with a controlled worker boundary. No scientific model, GPU, socket transport or real resident-process cleanup is certified by those probes. The delegation probe substitutes post-response resident observations because no real resident process exists. The post-compute boundary unit test advances State's clock deterministically and is not a performance measurement.

Entry bookkeeping locks and cleanup remain blocking. A call can return after its timeout; the fix charges that wait before deciding whether dispatch is allowed. It deliberately does not change lock ownership or request-ID deduplication to claim a hard wall-clock return bound. Request IDs remain consumed as before, expired admission releases reservations, and post-dispatch OutcomeUnknown still fences the device and preserves the no-replay policy.

Raw evidence is kept in `deadline-before/`, `deadline-after/`, `delegated-before.json`, `delegated-after.json`, `entry-lock-before.json`, `entry-lock-after.json`, red/green test logs and JUnit files, and `regression-pythonpath.json`. Probe outputs include the source commit and/or exact broker source hash; outputs measured before committing explicitly retain their source patch/hash. The entry-lock after-probe preceded only the final comment wording in eac3081; its functional code is unchanged. The coordinator and Agent C independently reproduce relevant findings before integration.

No throughput improvement is claimed here. The remote CPU workflow is the authorized environment for the frozen same-runner comparison; only its complete 30-trial raw results may support a performance conclusion. No GPU resource has been requested or used.
