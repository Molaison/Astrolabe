# Agent B independent interrogate: CandidatePoolScheduler pilot

**Verdict: PASS within the frozen CPU identity-safety scope.** No reproducible defect introduced by this patch was found. This is an independent verifier result from Agent B, who did not author C's patch.

## Exact provenance

- Baseline: `d66ee714f2e755948534f95d9d125287ce8dbd91` in `../agent-b`.
- Reviewed HEAD: `d57d2ef3c5bcf442228eb71f4f960819faac76b5` in detached `../agent-b-pilot-review`.
- Production diff: 11 lines in `astrolabe/experimental/candidate_batch_policy.py::CandidatePoolScheduler._run_task`; the associated test file adds 122 lines. No other production owner changed.
- Review worktree was clean after validation. No production edits or commits by this verifier.
- Runtime: shared CPU venv; actual Python ThreadPoolExecutor and scheduler. Worker results are synthetic identity-fault inputs. No pretrained model, GPU, broker socket, or GPU acceleration was tested.

## Independently repeated evidence

| Check | Baseline | Pilot |
|---|---|---|
| Four incorrect `(candidate_id, generation)` cases, both failure policies, both collection entry points: 16 fault gates | Gate failed; command exit 1 | All 16 passed; command exit 0 |
| Same-generation foreign candidate with high score | Can be selected as foreign candidate | Recorded only as failed submitted task; no score or payload propagated |
| Previous/future-generation response | Can populate that generation's ready results | Never injected under the returned generation |
| `fail-fast` via collect and drain | Does not reject forged completed result | Raises `CandidateExecutionError` identifying submitted task; ready list has no forged result |
| Legitimate returned CandidateResult and scalar score | Pass | Pass; exact raw normal-case JSON equals baseline |
| Worker RNG, score, input payload | One seeded local RNG draw; captured baseline | Exact RNG state, score, and input payload match |
| Legitimate returned object and payload identity; positive duration | Preserved | Preserved |
| Policy + runner + execution identity pytest suites | Not rerun for this report | **45 passed in 1.43s** |

Raw files: `pilot-identity-baseline.json`, `pilot-identity-fixed.json`, `pilot-verifier-tests.txt`. The independent probe is `pilot_identity_probe.py`. Expected identity errors are emitted by the existing event logger on stderr; they are not test failures.

The repair validates identity before assigning `result = raw`. Its existing exception path returns a new failed result with `task=task`, no score and no payload; both `collect_until_ready` and `drain` therefore consume the same normalized result. Legal worker output and the scalar normalization path are unchanged. Existing score selection, generation carry behavior, and runner semantics were also exercised by the 45-test suite. This does not establish MCMC scientific numerical equivalence; the pilot is an experimental candidate scheduler safety fix and modifies no acceptance/RNG implementation.

## Reproduction

From `cloud-20261104/agent-b`:

```bash
../venv/bin/python evidence/runtime-next/agent-b/pilot_identity_probe.py . --require-fixed > evidence/runtime-next/agent-b/pilot-identity-baseline.json
# expected exit 1 at baseline d66: fault gates expose the bug
../venv/bin/python evidence/runtime-next/agent-b/pilot_identity_probe.py ../agent-b-pilot-review --require-fixed > evidence/runtime-next/agent-b/pilot-identity-fixed.json
# expected exit 0 at reviewed d57
```

From `cloud-20261104/agent-b-pilot-review`:

```bash
../venv/bin/python -m pytest -q tests/test_candidate_batch_policy.py tests/test_candidate_batch_runner.py tests/test_execution_identity.py
```

## Limits and handoff

This PASS covers reply identity binding, its existing failure policy, legitimate payload/RNG noninterference, and the named CPU suites. Real AF_UNIX broker E2E, process PID liveness checks, GPU execution and pretrained output are separate blocked/untested scopes in this environment. Parent's final integration must rerun the frozen tests at the final integrated HEAD. The evaluator investigation remains in `investigation.md`, ready for the separately authorized ESMFold boundary work.
