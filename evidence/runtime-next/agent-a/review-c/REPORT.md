# Independent Agent A review of Producer submission fixes

**Verdict: PASS for the reviewed scope at `8c987ed86452d2c1bb467a1d20c2434347b5a95b`.** No reproducible new defect was retained. This does not certify a bounded queue, durable recovery, arbitrary streaming producers, concurrent submit callers, or a real model/GPU backend.

## Intent and exact scope

Review `754976e0443a84e3bcafe8f30bd8017dc6844a76` and `8c987ed86452d2c1bb467a1d20c2434347b5a95b` against the accepted identity pilot `d57d2ef3c5bcf442228eb71f4f960819faac76b5`. The intended behavior is to validate a complete finite fixed-parent generation plan before any worker starts and reject previously submitted `(generation, candidate_id)` without silently replaying scientific work. Preserve valid task count, parent content, RNG behavior, scores, selection and failure policies.

Two separate detached reviewer worktrees were created: `agent-a-producer-before` at d57d2ef and `agent-a-producer-review` at 8c987ed. The author's source worktree was read only. The same author probe files were copied unchanged into these isolated worktrees; output was redirected into this reviewer evidence directory.

## Independent executions

| Gate | Before at d57 | After at 8c987ed |
|---|---|---|
| Full updated policy tests | 10 failed, 16 passed | Included in 57 passing policy/identity/runner tests |
| Mixed-generation finite batch | Two worker calls; no rejection | Zero worker calls; explicit ValueError |
| Duplicate identity within batch | Two calls under one identity | Zero worker calls; explicit ValueError |
| Replay after collection | Two total worker calls | One total call; replay rejected |
| Valid two-candidate control | Two calls, selected score 0.75 | Identical calls and selected score |
| Same candidate ID in different generations | Two calls, one per generation | Identical behavior |

The new tests failed against the unchanged pilot source in the expected ten invalid-input/replay cases. Their accepted controls remained green. After source review, the independent reviewer ran `tests/test_candidate_batch_policy.py tests/test_execution_identity.py tests/test_candidate_batch_runner.py` at the exact final commit: **57 passed, 0 skipped**. Tests include invalid first/last task, an iterator failing after a valid prefix, retry of a previously rejected batch, replay while running/after collection/after failure, reconstructed result identity, and worker RNG/data preservation.

### Caller and scientific boundaries

`CandidateDecisionRun.run` submits `_tasks_for_generation(generation)`. That owner already constructs a finite list with monotonically distinct `cand_NNNN` IDs and copied `current_sequences` for the fixed parent before `submit_generation` is entered. Materializing and validating that supplied plan does not call the mutator, add candidates, or advance acceptance/RNG itself. The existing executor submit loop, result normalization, fail policies and selector are unchanged after validation.

The review does not extend this claim to arbitrary lazy generators whose generation work depends on concurrently executing workers: the revised interface explicitly validates a finite plan before execution. Executor failure midway through dispatch remains a different boundary from invalid input-plan rejection; the report correctly does not claim transactional rollback of work already submitted to an executor.

## Independently retained negative result

`probe_blocking_backpressure.py` was run at the reviewed commit. The actual event-controlled scheduler executes three tasks and completes three results on both sides. Current submission yields first selection **candidate 0 / score 1.0 / one result**, while a capacity-one blocking submit wrapper yields **candidate 1 / score 2.0 / two results**. Thus the prototype changes the first collection and selected parent even without changing total task count. Rejecting it is justified; no such production change appears in the diff.

The independent queue probe also reproduces the remaining gap: `max_in_flight=1` gives **8 unfinished tasks, 1 running and 7 waiting**, followed by 8 calls and selected score 7.0 on drain. The setting remains a worker count, not a fixed bound on queued/uncollected work. The submitted-identity set also grows with admitted work. These limits are explicitly PARTIAL rather than silently described as solved.

## Findings and limits

Retained findings: **0**. Additional theoretical claims were not promoted into bugs. In particular, out-of-contract concurrent submit callers and infinite producer iterators were excluded from the fixed-parent production claim; no production caller exercising those patterns was found in the reviewed call chain. Existing resource/backpressure limitations remain documented above.

This is scientific control-plane/producer-boundary evidence using inexpensive deterministic workers. It does not make a performance, GPU, model residency, MCMC-chain concurrency or durable exactly-once claim. The final integrated HEAD still needs the coordinator's complete end-to-end gate.

## Reproduction and artifacts

From the respective reviewer worktree:

```sh
../venv/bin/python evidence/runtime-next/agent-c/probe_submission_boundaries.py
../venv/bin/python evidence/runtime-next/agent-c/probe_blocking_backpressure.py
../venv/bin/python -m pytest -q tests/test_candidate_batch_policy.py tests/test_execution_identity.py tests/test_candidate_batch_runner.py
```

For the negative policy gate at the pilot baseline, the updated test file was copied as `evidence/runtime-next/agent-c/test_policy_review.py` and executed without changing production source.

Raw files in this directory: `submission-before.json`, `submission-after.json`, their stderr logs, `backpressure-negative.json` and stderr, `regression-before.log/xml`, `regression-after.log/xml`. Independent observations match the author's recorded failures and valid controls.
