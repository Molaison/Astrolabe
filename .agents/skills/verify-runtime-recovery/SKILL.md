---
name: verify-runtime-recovery
description: "Verify Astrolabe durable Runtime recovery after changes to call ordering, journal writes, operation budgets, or the shared prediction/Score recovery boundary."
---

# Verify Runtime recovery

An issued durable Runtime prediction or Score call must never run automatically a second time.
Every reserved call must keep consuming its budget after a crash.
A violation can duplicate expensive work or change the scientific search budget.

## Run the acceptance command

Use Linux, Java 8 or newer, and the checkout's Python 3.11 or newer development
environment. The real-code checks use `pytest`, `jsonschema`, and the repository
code. `python -m pip install -e '.[dev]'` prepares a fresh development environment.
No model, weight, GPU, or external inference service is used.

From the repository root, run:

```bash
python scripts/validation/verify_runtime_recovery.py --self-test
```

`just verify-recovery` runs the same command through `GH_PY`.
The runner downloads TLA+ 1.7.4's TLC 2.19 JAR and verifies its SHA256.
For an offline run, provide that same JAR with `--jar` or `TLA2TOOLS_JAR`.
Use `--output` with a new directory to retain logs and `result.json`.
With no output argument, the runner allocates a temporary directory and prints it.

Accept only an exit code of zero and `result.json` with `status` equal to `passed`
and `mode` equal to `self-test`.
The command requires both normal models to pass, all real-code tests to pass with
no skips, all deliberately broken models to violate their named invariant,
all real-code mutations to fail their corresponding test, and the original
code to pass again. A timeout, missing dependency, or stale coverage record fails
the command. Negative checks have nonzero child exits by design.

## Understand the scope

TLC enumerates the reachable states of
[`RuntimeRecovery.tla`](../../../verification/runtime_recovery/RuntimeRecovery.tla).
It checks two named call slots under one shared accounting boundary, with budgets
of one and two. One owner executes calls serially. Crashes may occur between
the modeled steps arbitrarily often. No fairness or eventual completion is claimed.

`AtMostOnce` bounds each slot's invocation count by one.
`BudgetNeverRefunded` bounds the number of reserved or later journal entries by
the configured budget. A reserved call consumes budget even if never invoked.
`IssuedBeforeInvoke` requires observed work to retain an issued or terminal
journal state.

The model treats a successful `Runtime._write` as one SQLite transaction.
It assumes committed rows survive process death, the owner lock excludes other
processes, the request/component binding stays fixed, and no external writer
edits the journal. It does not model concurrent threads calling one Runtime,
disk corruption, power loss, owner-verified `reconcile`, arbitrary request data,
RNG equivalence, ScoreReceipt contents, member commits, GPU completion, or
model numerical correctness.

The real-code test forks a process and uses `os._exit` before and after the
reserved, issued, and completed writes, after an observable invocation, and on
the normal path. It opens the actual SQLite journal on recovery and retries
twice. It runs these cuts for both `predict` and `score` accounting labels.
These are representative implementation checks, not exhaustive Python verification.

## Review the model against the code

The verified scope is the journal decision within
[`Runtime.call`](../../../astrolabe/design/runtime.py), not every function in
its dependency files.

| Model action | Actual code boundary |
| --- | --- |
| `Reserve` | `Runtime.call` budget check and `_write(... status="reserved")` |
| `ResumeReserved` | Existing matching reserved row reaches the issue path |
| `Issue` | `_write(... status="issued")` before `operation.invoke` |
| `Invoke`, `Return` | Entry to and return from `FunctionOperation.invoke` |
| `Complete` | Result and RNG persistence precede `_write(... status="completed")` |
| `Fail` | Runtime records `unknown` or owner-certified `failed` |
| `Crash` | Process exits and releases its owner lock; committed rows remain |
| `ObserveIssued` | A repeated issued call records unknown and raises `OutcomeUnknown` |
| `Replay` | Deliberately broken variant only; normal configuration disables it |

[`coverage.json`](../../../verification/runtime_recovery/coverage.json)
records the sync commit and SHA256 values of `runtime.py`, `operations.py`,
`store.py`, `types.py`, and `model_execution.py`.
The last code review anchor is
`46501fc12136c915516d9a39455767c6f9dbe48b`.
Dependency hashes trigger review; they do not claim that all their functions
are modeled. The runner refuses stale hashes.

## Diagnose a failure

Read the failed child log linked in `result.json`.
A TLC invariant violation includes the state sequence. Translate it into the
table above before deciding whether the implementation or model is wrong.

The no-replay counterexample is an invocation, an unknown outcome, then another
invocation of the same slot after recovery. The real-code mutation test obtains
that unknown outcome by terminating the process before completion is recorded.
The budget counterexample leaves the first slot unknown and admits another
slot after incorrectly refunding its reservation.

When the model passes but real-code tests fail, inspect the journal status and
the `effects` count in the pytest failure. Check the model's atomic boundaries
against `Runtime._write` and `Runtime.call`. Do not weaken the invariant to
make the command pass. Report a product defect without changing product code
unless that repair is authorized.

The runner's `--self-test` changes temporary copies of the real
`astrolabe/design/runtime.py`. It separately disables the issued/unknown replay
guard, counts only completed attempts against budget, and removes the issued
journal write. It checks the
imported source path, requires behavioral pytest failures, deletes each copy,
and reruns the unchanged source. A checksum failure alone is not a successful
mutation check.

## Keep the verification synchronized

On a covered change, first compare the new code with the recorded sync commit
and the model-action table. Inspect the new ordering and transaction boundaries.
If intended behavior changes, update the model and the real-code checks together.

During diagnosis, run:

```bash
python scripts/validation/verify_runtime_recovery.py --behavior-only
```

That bypasses the source-sync gate and runs only real-code checks. It is not the
formal acceptance command.

Only after checking the correspondence, update the reviewed files' hashes and
advance the sync commit in `coverage.json` to a commit that contains the reviewed
implementation. Update this skill's review anchor in the same change. Never
refresh hashes automatically to silence the gate. Run `--self-test` again.
Use `/maintain-formal-verification-skill` when intervening code changes were not
checked. Keep findings open when the current implementation still violates them.
