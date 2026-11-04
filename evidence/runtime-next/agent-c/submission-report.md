# Agent C swarm: producer submission boundaries

**DONE:** two narrowly scoped correctness fixes, separate commits, baseline failures preserved and targeted regression gates passing. **PARTIAL:** the experimental candidate pool still has no fixed bound on queued/uncollected work. A naive blocking admission design was rejected by an executable scientific-selection counterexample. Independent review/integration remains with the coordinator.

## Commits and ownership

Both commits follow Agent C's original identity pilot `d57d2ef3c5bcf442228eb71f4f960819faac76b5`, which the coordinator independently reviewed and integrated as code-equivalent `932de1b`.

| Commit | Change | Regression result |
|---|---|---|
| `754976e0443a84e3bcafe8f30bd8017dc6844a76` | Validate the complete finite generation batch before dispatch | New gate: 5 failed / 1 passed before; full relevant suite: 51 passed after |
| `8c987ed86452d2c1bb467a1d20c2434347b5a95b` | Reject duplicate `(generation, candidate_id)` within a batch or previously submitted to the same scheduler | New gate: 5 failed / 1 passed before; full relevant suite: 57 passed after |

Only `astrolabe/experimental/candidate_batch_policy.py` and `tests/test_candidate_batch_policy.py` changed in those commits. No push. No changes to `run.py`, acceptance, RNG, model code, or another worktree. The preexisting untracked framework-homology file remains untouched.

## Verified defects and outcomes

`probe_submission_boundaries.py` calls the actual experimental ThreadPool scheduler with inexpensive deterministic workers. It is a boundary fault injection, not a model/GPU benchmark.

| Scenario | Baseline at d57 | After generation validation | After duplicate validation |
|---|---:|---:|---:|
| Two unique candidates, declared generation matches | 2 worker calls; best score 0.75 | Same | Same |
| Generation 1 batch includes generation 2 task | 2 worker calls, no error | 0 worker calls, explicit ValueError | Same |
| Same ID/generation occurs twice in one batch | 2 worker calls under one identity | Still 2 | 0 worker calls, explicit ValueError |
| Same task resubmitted after collection | 2 total worker calls | Still 2 | 1 total call; replay rejected |
| Same candidate ID in distinct generations | 2 worker calls, one per generation | Same | Same |

The full regression gate also proves invalid tasks in either the first or last position cannot dispatch a valid prefix, a producer iterator failure cannot dispatch its yielded prefix, and a rejected batch does not reserve fresh task identities. Replay is rejected while an original task is running, after collection, and after an explicit worker failure. A failure alone is not authorization to rerun the same scientific task. Callers can explicitly create a new identity for a new task.

Input validation errors are `ValueError` before any worker starts. Runtime worker failures retain the existing `record-and-continue` / `fail-fast` behavior. The scheduler retains accepted task identities for its lifetime so collecting or failing a result cannot erase the no-replay guard. This is local, in-memory admission state; it is not a durable checkpoint or cross-process exactly-once protocol.

The current production `CandidateDecisionRun._tasks_for_generation` already constructs finite fixed-parent lists. Validation adds no task, drops no valid task, changes no score, and keeps the existing executor dispatch loop and selection function. The API now materializes an iterable before dispatch so a later validation/iteration error cannot start partial scientific work. Arbitrary streaming/infinite producers are not a supported bounded queue.

## Backpressure: verified gap and rejected alternative

With `num_candidates=8` and `max_in_flight=1`, an event-controlled worker probe observes **8 unfinished submitted tasks: 1 running, 7 waiting**. After release, all 8 run and the full drain selects score 7.0. This observation is unchanged by both correctness fixes. `max_in_flight` currently means executor worker count; there is no separate fixed queue cap.

A second script, `probe_blocking_backpressure.py`, tests a deliberately narrow alternative on a scheduler instance: block `executor.submit` until a capacity-1 slot is free. The same three deterministic tasks, scores and total work are used. Explicit events eliminate timing guesses:

| Protocol observation | Current submit behavior | Capacity-1 blocking submit prototype |
|---|---:|---:|
| Total worker calls / completed tasks | 3 / 3 | 3 / 3 |
| Results in first collection | 1 | 2 |
| First selected candidate / score | ID 0 / 1.0 | ID 1 / 2.0 |

The producer cannot finish submitting the third task until the second releases capacity, so by the time the caller starts collecting, two results are already available. The existing best-of-first-available collection chooses a different parent. Therefore this simple backpressure change is **rejected**, despite keeping the total work budget. No production queue or blocking admission code was added.

A finite nonblocking capacity rejection would preserve explicit failure but could abort otherwise valid configured multi-generation work when old stragglers occupy the pool; silently dropping candidates would change the budget. A complete bounded design needs an explicit relationship between pending generation plans, admission and collection cutoffs. That is a remaining R2/R4 design/integration task, not a completed optimization. The existing independent-repeat scheduler is a separate already implemented concurrency boundary.

## Commands and raw evidence

Run from the checkout containing this evidence directory; for an independent checkout, copy the scripts into that checkout's equivalent evidence path so their import root matches the reviewed code:

```sh
../venv/bin/python evidence/runtime-next/agent-c/probe_submission_boundaries.py
../venv/bin/python evidence/runtime-next/agent-c/probe_blocking_backpressure.py
../venv/bin/python -m pytest -q tests/test_candidate_batch_policy.py tests/test_execution_identity.py tests/test_candidate_batch_runner.py
```

Raw data: `submission-before.json`, `submission-after-generation.json`, `submission-after.json`; complete red/green logs `generation-regression-{before,after}.log` and `duplicate-regression-{before,after}.log`; rejected design evidence `blocking-backpressure-negative.json`; environment/source and harness hashes in `submission-manifest.json`; decisions appended to `decision.tsv`. Earlier pilot evidence is preserved.

Final targeted suite on the code committed as `8c987ed`: **57 passed in 1.31 s**. `git diff --check` passes. Timing is reported solely as test-run metadata, not throughput evidence. No real pretrained GPU backend was available or used.

## Remaining limits / review handoff

The task identity check covers candidate ID and generation only; it does not add parent-state version, stage epoch, RNG state, constraint/model fingerprint, deduplicated durable commit or restart recovery. The scheduler remains single-caller owned; this change does not make concurrent `submit_generation` callers safe. Atomicity here covers invalid/incomplete input plans, not process failure or executor failure midway through actual dispatch. The submitted-identity set grows with admitted work, preserving replay protection for the pool lifetime.

Independent review should replay both old baseline failures and fixed outputs, verify the valid controls and original 45-test pilot/runner suite remain green, and retain the backpressure counterexample as a negative result. Agent C is available for the requested read-only ESMFold helper cross-review once Agent B's commit SHA is supplied.
