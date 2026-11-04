# Candidate result identity pilot

**DONE: implementation and local correctness gates. Independent review/integration pending.**

Commit: `d57d2ef3c5bcf442228eb71f4f960819faac76b5` (`fix: reject foreign candidate result identities`). Parent: `d66ee714f2e755948534f95d9d125287ce8dbd91`. No push.

## Hypothesis and bounded change

The submitted task is the authoritative identity. Validating a returned `CandidateResult` against its submitted candidate ID and generation before admitting its score/payload prevents foreign candidates and stale/future generations from entering the ready collection.

The commit adds 11 production lines in `astrolabe/experimental/candidate_batch_policy.py` and a focused test file `tests/test_candidate_batch_policy.py`. Identity mismatch raises inside the existing `_run_task` exception boundary; that boundary constructs a failed result using the original submitted task, with no score or returned payload. Existing `record-and-continue` (the skip-failed behavior) and `fail-fast` policies continue to decide collection behavior. No new failure policy, scheduling behavior, task fields, acceptance rule, model operation, or RNG transfer was introduced.

## Frozen reproduction and evidence

Run from the agent-c worktree:

```sh
../venv/bin/python evidence/runtime-next/agent-c/probe_candidate_identity.py
../venv/bin/python -m pytest -q tests/test_candidate_batch_policy.py tests/test_execution_identity.py tests/test_candidate_batch_runner.py
```

| Observation | Before d66 | After d57d2ef |
|---|---:|---:|
| Valid identity is selected | true | true |
| Wrong candidate is selected | true | false |
| Unsubmitted future generation is selected | true | false |
| New focused regressions | 12 failed / 2 passed | 14 passed |
| Existing identity and runner tests | 31 passed | 31 passed |

The focused tests cover wrong candidate ID, past generation, future generation, combined identity mismatch, original-task failure attribution, fail-fast via both collect and drain, and valid scalar/reconstructed-result paths. Valid-result controls retain exact score, returned payload object, caller payload, explicit duration and worker RNG progression. The full targeted gate is **45 passed in 1.30 s**. No performance conclusion is drawn from pytest timing.

Preserved files:

- `pilot-identity-before.json` and `pilot-identity-after.json`: unchanged three-case fault-injection harness output, with source HEAD.
- `pilot-identity-after.stderr.log`: expected diagnostic events for rejected wrong replies.
- `pilot-regression-before.log`: all 12 baseline failures, including missing fail-fast behavior.
- `pilot-regression-after.log`: 45 passing targeted checks.
- `pilot-manifest.json`: code SHA256 values, probe hash, environment, before/after counts and decision.
- `decision.tsv`: single kept hypothesis; negative baseline remains available.

`git diff --check` passed. The two committed files are the only tracked changes. The preexisting untracked framework-homology file remains untouched; evidence files remain unstaged for coordinator aggregation.

## Limits and review handoff

This verifies the experimental pool's returned `(candidate_id, generation)` boundary. It does not establish a complete trajectory/version/RNG/constraint fingerprint protocol, task deduplication, checkpoint recovery, real GPU throughput, or complete R2/R4/R9/R10 coverage. Built-in process workers already return the original task; the baseline failure is induced at the extension/failure boundary. The default MCMC loop remains unchanged.

Root independently reproduced the baseline before authorizing this implementation. Agent B/root should independently run the frozen probe and regression suite on this commit, then decide integration. Suggested independent adversarial checks are wrong identity with a failed status, matching identity reconstructed after transport, and a valid task alongside a foreign high-score reply. Source changes remain paused pending that review and the next module assignment.
