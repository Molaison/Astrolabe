# Runtime benchmark contract, 2026-11-04

## Outcome and fixed baseline

Preserve proposal, scoring, acceptance, stage transitions, seeds, input sizes and evaluation budget while improving useful completed work per wall second. GPU activity is diagnostic only. This contract is frozen before product-code changes.

The target branch was read through GitHub and Git at `d66ee714f2e755948534f95d9d125287ce8dbd91`, parent `834402daade9f62e9dec0b08041dade163a89369`. PR #8 is a draft with this head; main is `2a8b72a12550d8def971fe910fccf7d6d5f25dc0`. New work is stacked on #8. Neither main nor other branches are rewritten. `261b240` is an ancestor of this branch, but that does not prove the uncommitted server worktree was incorporated.

The old Repository Mental Model and Agent Context were recovered from their original standalone attachments. Their source baseline is `b537b8a81ff0738922e80a997ffeb9700cdafe09`; they are constraints and navigation, not evidence that today's runtime is unchanged. The runtime report and raw evidence already in `evidence/runtime-20261103` are reused without relabeling them as new measurements.

## Workloads and measurement

| ID | Frozen work | Required evidence | Current resource status |
| --- | --- | --- | --- |
| W0 | Existing `scripts/validation/fixtures/runtime_cpu.yaml`; Dummy evaluator, actual MCMC, 3 independent repeats, seeds 17/18/19, 3 steps per repeat | 9 actual evaluator calls per trial; proposal/score/decision and trajectory identities equal; complete provenance and resource release | Direct CPU runnable locally; original broker requires an ordinary Linux runner |
| W1 | One implemented pretrained evaluator plus its actual mutator, preserving model precision, recycles, samples and seed settings | Exact checkpoint/config checksums, numerical and decision parity, cold/steady timing, restart/OOM handling | BLOCKED: no GPU, weights, allocation or connected server executor in this session |
| W2 | The same existing independent repeat budget, allowing overlap only between trajectories | Bounded preparation and device admission; per-trajectory sequential MCMC; ordered output/unique completion; fault controls | CPU mechanics can be verified; GPU overlap needs W1 resources |

W0 scientific projection SHA-256 is `f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5`, identical to the archived comparison. The existing harness SHA-256 at freeze is `8873b26f63a0d232bdae672c826fba1d7506fa6e4bbc48a9708dcec91e7b1094`; fixture SHA-256 is `d4d757775fb3948394f82f59150e85a365d2683091eb11f68f1725253b537dd9`.

The original three-mode plan is five groups, 15 trials and 45 trajectories. For a before/after code comparison, execute both the fixed baseline and candidate in every mode: five groups of six trials, 30 trials and 90 trajectories total, with a 900-second total CPU wall-time ceiling and 260 MB output ceiling. Rotate the modes and alternate baseline/candidate order within the same runner. The work in each trial stays 3 repeats and 9 evaluations. No uncounted scientific warmup or tuning samples. Record all failed attempts. Profiles are separate from timed measurements. This comparison schedule was chosen before any product-code change.

The frozen primary metric is successful scientifically valid evaluations per total workflow wall second, including startup and shutdown. Report accepted mutation steps separately from initialization; do not equate accepted progress and evaluation throughput. Report median, range, paired differences and uncertainty. CPU engineering target is at least 20% lower overhead on W0, beyond run variation, with no scientific difference. A fast path must also remain within 10% of the same environment's legacy direct workflow. These are acceptance targets, not observed results.

If a proposed optimization cannot be measured in an eligible environment, label its performance unverified. Never divide a successful trial by the time of a failed trial, compare different machines as a speedup, or treat skipping a broker as proving model residency performance. The historical 9.691/32.418/33.455-second medians remain historical evidence only.

## Scientific and recovery gates

Keep single-trajectory proposals causally downstream of the previous evaluation and acceptance. Keep stage and accepted-state writers in their current owner. Independent repeats use process isolation; no shared `global_config` in trajectory threads. Compare proposals, all scientific score rows, acceptance/gates, stage transitions and terminal provenance. Unknown outcomes are not automatically retried; stale or duplicate completions cannot be committed.

Use existing regression tests at the affected boundaries plus focused reproductions of actual defects. Test doubles certify the contract they exercise, never a pretrained backend. No full checkpoint/resume claim is allowed without a complete-state replay. Missing output fields stay unavailable rather than becoming zeros.

## Execution environment limits

The local interpreter is Python 3.12 in an isolated virtual environment, with CPU-only execution. Unix sockets are denied and the process API PID differs from `/proc`; existing runtime safety checks are preserved. Direct W0 can run; broker/guardian end-to-end tests need an ordinary Linux environment. GitHub Actions is an available candidate CPU runner, pending an actual completed run. The existing workflow is active but no prior runs were returned at preflight.

Subsequent execution evidence: GitHub run `41760526028` at candidate `6620da607e66d569eed802057d77b0fbfa9c37b4` passed the real environment gate, 242 regression tests with no skips, and all 30 fixed W0 trials. The complete 645-file raw artifact is preserved losslessly as `evidence/runtime-next/ci/runtime-cpu-6620da6.tar.xz`; its small derived summary is `cpu-ci-01-summary.json`. This establishes an eligible CPU runner without weakening the local guards. It does not establish GPU availability. No meaningful performance gain occurred in that correctness-only comparison. All 180 mutation attempts were rejected; the 90 accepted initialization rows are not accepted mutation progress. An additional, separately frozen correctness control may cover acceptance and stage transitions, but its work must never be counted as W0 timing or acceleration.

The requested `/xcf26/zpzeng/practice_arena/astrolabe_runtime_20261102_implementation/cloud-20261104` is not mounted. No SSH configuration is available; the connected native execution service reports disconnected. Destination transfer and GPU validation stay BLOCKED. Do not manufacture a local directory with that server path or claim successful remote writes.

## Pilot and subsequent rounds

The first pilot closes the experimentally reproduced candidate-result identity defect. Both Agent C and the lead independently observed a worker reply with an unsubmitted candidate or future generation being selected by the real experimental pool. This narrow correctness pilot can be fully verified in the available environment before performance work fans out. It carries no speed claim. The CPU regression remains R1, with a new constraint from the investigation: bypassing the broker would drop default resource monitoring even when no explicit resource overrides are configured. That shortcut is rejected.

Each code owner and independent reviewer must record the exact source tree, behavioral evidence and keep/revert decision. Subsequent performance attempts get separate rows in `evidence/runtime-next/decision.tsv`; failed or inconclusive attempts remain in the evidence. Correctness fixes require before/after failing reproductions and do not carry speed claims.
