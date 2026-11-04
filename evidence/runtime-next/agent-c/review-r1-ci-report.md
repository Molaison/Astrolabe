# Independent review of R1 W0 CI artifacts

**Review status: DONE.** Agent C independently reanalyzed the complete persisted CPU CI artifact set for candidate `71c7592e27d02b2772d984d9b1f5ed9fea47fd6c` versus baseline `d66ee714f2e755948534f95d9d125287ce8dbd91`. All 30 planned trials, 90 trajectories and 270 evaluator calls are present and scientifically identical for the corresponding repeat/seed. Every same-mode pair favors the candidate. **The original 20% target remains unmet, and resident modes remain slower than direct.** This is a CPU/Dummy result, not pretrained GPU evidence.

## Independent method

`review_r1_ci.py` reads all 90 original CSV/provenance/config sets, all 30 trial configs/runtime traces/telemetry streams, the frozen plan, and individual process-observer records. It does not import the benchmark controller or use its aggregate performance, correctness, embedded scientific rows or summary conclusions. No benchmark, model, production code or scientific workload was run or changed during this review.

The script hashes all **645 original artifact files** before and after analysis. It verifies **433 baseline and 436 candidate source-file hashes** against their exact Git revisions, resolving the two recorded configuration symlinks to their in-repository target bytes. The four measurement sources and the comparison wrapper match their frozen hashes. Individual elapsed measurements are cross-checked using each raw telemetry stream's inferred monotonic origin and the persisted process-completion endpoint; maximum disagreement is **1.372 microseconds**. This reanalysis shares the original observer's approximately 30 ms completion-observation resolution and is not a new independent timing experiment.

Results and every trial's input identities, elapsed measurement and CSV hashes are in `review-r1-ci-02/report.json`; the complete raw-file inventory is `review-r1-ci-02/input-manifest.json`.

## Work and scientific consistency

| Independently counted item | Result |
|---|---:|
| Trials | 30: five interleaved groups × two revisions × three modes |
| Completed trajectories | 90: repeat/seed pairs `(0,17)`, `(1,18)`, `(2,19)` in every trial |
| Actual evaluator calls | 270; each trial has exactly nine of each CPU prepare/execute/finalize phase |
| Scientific score rows | 270; steps 0, 1 and 2 in every trajectory |
| Initial rows / accepted initial rows | 90 / 90 |
| Mutation attempts / accepted mutations | 180 / **0** |
| Accepted improving mutations / stage-entry rows | **0 / 0** |
| Process releases / empty terminal broker ledgers | 30 / 20 |

All columns and raw CSV bytes agree exactly for each repeat/seed across all modes, groups and revisions. The initial sequence is `AAAA`, score 1.0. Each mutation has score 0.75 and is rejected; the sequences are `GAAA`, `SAAA` for seed 17; `PAAA`, `TAAA` for seed 18; and the exact seed-19 records are retained in the machine-readable report. The initial accepted row is not a successful mutation or evidence of scientific progress. Completed-trajectory throughput measures completion of this fixed protocol, not productive accepted mutations.

The entire trial configuration is checked against the frozen configuration with only explicit mode controls and output paths applied. All 90 executed configurations also agree after validating and removing their specific repeat/seed and output paths and normalizing the three known execution-mode fields. No scientific field differs. The complete configuration, protocol hash, measurement hashes and budgets also match earlier raw plans `cpu-ci-01` and `cpu-ci-02`.

Protocol SHA-256 remains `f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5`. Mode order rotates by group, baseline/candidate order alternates AB/BA, no extra trial directory exists, and observed trials do not overlap. Each trial starts a fresh CLI. Hardware, Python and dependency provenance agree; GPU allocation/visibility is empty.

## Recomputed performance

Ratios pair the baseline and candidate within the same group and mode. Throughput improvement is `baseline_seconds / candidate_seconds - 1`; elapsed-time reduction uses the corresponding reciprocal and is a different percentage.

| Mode | Baseline median seconds [range] | Candidate median seconds [range] | Median paired throughput improvement | Median paired time reduction | Faster pairs |
|---|---:|---:|---:|---:|---:|
| direct | 2.0261 [2.0186–2.2950] | 1.8798 [1.8484–1.9530] | +9.27% | 8.49% | 5 / 5 |
| resident_serial | 5.4685 [5.3690–5.5285] | 4.7865 [4.7416–4.8503] | +13.98% | 12.27% | 5 / 5 |
| resident_adaptive | 5.4677 [5.3627–5.5632] | 4.7920 [4.7309–4.9220] | +12.56% | 11.16% | 5 / 5 |

All three observed ranges separate strictly: the slowest candidate is faster than the fastest baseline in that mode. Separation margins are 65.60 ms, 518.65 ms and 440.73 ms respectively. This is repeatable improvement within this five-group experiment; it is not a confidence bound or a claim about other machines, workloads or long resident sessions. Individual pairs remain visible, including the direct mode's first +24.16% pair; selecting that pair alone would overstate the typical result.

The best direct revision by median is the candidate. Against its direct mode, candidate serial and adaptive resident throughput ratios have paired medians **0.3941×** and **0.3923×**. Resident elapsed times are **2.537×** and **2.549×** direct, with **0/5** resident-faster pairs. Therefore resident has not reached parity with direct, let alone 20% higher throughput. Even the three same-mode paired medians remain below 20% improvement.

## Preserved limits and reader correction

This experiment includes cold CLI startup and shutdown. The candidate contains R1 and earlier validated changes; this review does not isolate the causal contribution of each commit. All results are CPU/Dummy and have no model weights, GPU execution or structure output. W0 does not snapshot the complete RNG or checkpoint state and has no accepted mutation/stage transition, so the separate CLI scientific positive control remains necessary and is not merged into these performance samples.

The initial offline reader omitted symlink members from a Git archive and stopped at `config/env/env.yaml` before completing verification. That reader, its error and the unchanged input inventory are retained in `review-r1-ci/`. The corrected reader resolves links inside the recorded archive without following host filesystem links; its full audit passes in a new output directory. This is a reader-development correction, not another benchmark attempt or a scientific/performance failure.

## Reproduction

From the Agent C checkout, with the original CI artifacts available and both Git objects present:

```sh
../venv/bin/python evidence/runtime-next/agent-c/review_r1_ci.py --artifacts ../control/cpu-ci-03 --repo . --prior ../control/cpu-ci-01 --prior ../control/cpu-ci-02 --out evidence/runtime-next/agent-c/NEW-r1-review
```

The script requires a fresh output directory outside the raw artifact tree. It writes only its own report and file inventory. The raw archive is also preserved by the coordinator as `evidence/runtime-next/ci/runtime-cpu-71c7592.tar.xz`; directory relocation does not affect individual file hashes or scientific comparisons.
