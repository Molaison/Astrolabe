# Runtime 2026-11-04 evidence index

Read the [final report](../../docs/runtime_final_report.md), [frozen contract](../../docs/runtime_benchmark_contract.md) and [architecture decision](../../docs/runtime_architecture_decision.md) first. Overall delivery is **PARTIAL**. The fixed CPU throughput target and final-HEAD scientific/contract gates are verified. The real requested server destination is delivered. Pretrained GPU execution is **BLOCKED by actual Slurm admission**, with the full failed request and zero allocated GPU-hours preserved.

**Final source HEAD is `1bdaf3a53227c9197b930dc00a01b752acd90067`.** See the [current final validation record](final-head-1bdaf3a/README.md): 378 CPU +181 geometry/Torch tests, zero skips/failures, complete 30-trial CPU comparison, independently rerun acceptance/stage/RNG control, complete raw archives and source/production equivalence. The [earlier 9d validation record](final/README.md) is retained as historical evidence. This evidence branch adds only docs/data while preserving the tested PR source.

## Production and review anchors

| Evidence | Entry point |
|---|---|
| Initial Runtime investigation, historical trace decomposition | [Agent A investigation](agent-a/INVESTIGATION.md), `agent-a/agent-a-raw.tar.xz` |
| Minimal pilot and producer integrity | [C pilot](agent-c/pilot-report.md), [C submission](agent-c/submission-report.md), [A independent C review](agent-a/review-c/REPORT.md), `review/probe_submission_root.py` |
| Admission deadline and delegation ownership | [A runtime correctness](agent-a/RUNTIME_CORRECTNESS.md), [C independent A review](agent-c/review-a-report.md), `review/deadline-{before,after}`, `review/delegation-*`, `review/entry-lock-*` |
| ESMFold audited output boundary | [B output report](agent-b/esmfold-output-report.md), [C independent B review](agent-c/review-b-report.md), `review/esmfold-*` |
| R1 cold import change | [A hypothesis/negatives](agent-a/R1_SPATIAL_IMPORT.md), [B independent R1 review](agent-b/review-a-r1-report.md), `review/spatial-root.*`, `review/r1-source-root.json`, `review/r1-missing-spatial-root.json` |
| R1 end-to-end performance | `ci/cpu-ci-03-summary.json`, `ci/runtime-cpu-71c7592.tar.xz`, [lead independent audit script](review/audit_w0_root.py), `review/r1-w0-root-audit.json`, [C independent raw review](agent-c/review-r1-ci-report.md) |
| R1 bounded follow-up / stopping decision | [A R1 round2](agent-a/r1-round2/REPORT.md); no further product edits, 0 scientific evaluations |
| Actual acceptance/stage/RNG positive control | [C report](agent-c/scientific-positive-report.md), C first failed count expectation and corrected freeze/run, `review/scientific-positive-root-summary.json`, `review/scientific-positive-root.tar.xz` |
| Complete registry/config scope inventory | [backend matrix](../../docs/runtime_backend_certification.md), `backend-certification-final.json`, `review/backend-matrix-root.*` |
| Measurement unchanged / environment limits | `review/frozen-source-root.json`, `review/resource-availability.json`, `review/harness-preflight/report.json` |
| Keep/reject and negative attempts | [decision.tsv](decision.tsv) plus each owner's report and raw manifest |

Paths in this file are relative to `evidence/runtime-next`.

`decision.tsv` is an attempt log: each status describes that historical attempt,
including initial resource/destination blockers later superseded by successful
delivery or a different bounded experiment. Use the final report for current
R0–R12 status. Every early attempt now has a concrete evidence path and known
source anchors; unrecorded total CPU/wall costs remain explicitly unknown.
Pre-commit dirty probes are interpreted with their recorded source/patch hashes,
not by treating their then-current HEAD as the tested patch. The original
`decision-legacy.tsv` remains unchanged.

## Actual CI runs and raw preservation

| Run / source | Scope and outcome | Lossless raw archive |
|---|---|---|
| 41759696271 / 68888f8 | Workflow parser rejected runner context; no benchmark | `ci/initial-attempts.json` |
| 41760013668 / 4358cc3 | 220 pass, 4 fixture-child import failures; no benchmark | `ci/first-regression-failure.log`, initial attempts metadata |
| [41760526028](https://github.com/Molaison/Astrolabe/actions/runs/41760526028) / 6620da6 | 242 tests, 30 trials; correctness passed, no credible speed gain | `ci/runtime-cpu-6620da6.tar.xz`, 645 raw files |
| [41762677554](https://github.com/Molaison/Astrolabe/actions/runs/41762677554) / db937f9 | 247 tests, 30 trials; correctness passed, no credible speed gain | `ci/runtime-cpu-db937f9.tar.xz`, 645 raw files |
| Same run / db937f9 | 65 CPU Torch tests, zero skips; synthetic checkpoints, no pretrained/GPU | `ci/runtime-torch-db937f9.tar.xz`, 14 raw files |
| [41764768462](https://github.com/Molaison/Astrolabe/actions/runs/41764768462) / 71c7592 | 248 tests, 30 valid trials, R1 CPU gain beyond observed variation | `ci/runtime-cpu-71c7592.tar.xz`, 645 raw files |
| Same run / 71c7592 | 104 CPU Torch/structural tests, zero skips | `ci/runtime-torch-71c7592.tar.xz`, 15 raw files |
| [41766956074](https://github.com/Molaison/Astrolabe/actions/runs/41766956074) / 9d62b3e | Historical final: 248 CPU +104 geometry; 30 valid trials | `ci/runtime-cpu-9d62b3e.tar.xz`, `ci/runtime-torch-9d62b3e.tar.xz` |
| [41780311942](https://github.com/Molaison/Astrolabe/actions/runs/41780311942) / **1bdaf3a** | **378 CPU +181 geometry; zero skips/failures; 30 valid trials**, all 15 pairs faster | `final-head-1bdaf3a/agent-c/final-head-c-raw-and-review.tar.xz`, original ZIPs plus all raw/artifact/audit files |

Each successful W0 comparison has 90 trajectories / 270 evaluations / 180 rejected mutation attempts. The 90 accepted initialization rows are not accepted mutation progress. Different CI runners' absolute timings are never compared as speedup. The separate science-control evaluations and 18 A diagnostic evaluations do not enter W0 timing. CPU tests use real CPU libraries and, where named, fake model weights; no row certifies pretrained GPU inference.

Archives preserve the extracted GitHub artifact bytes including install/freeze environment, commands/config, provenance/CSV, traces, telemetry, observer state, JUnit and reports. Summaries include exact artifact IDs and original ZIP SHA-256. CI artifacts also remain available on GitHub for their configured retention, but these archives avoid relying on that retention.

## Verify and extract

From this directory, check the archive manifest:

```bash
sha256sum -c archives.sha256
mkdir -p /absolute/empty-evidence
tar -xJf ci/runtime-cpu-71c7592.tar.xz -C /absolute/empty-evidence
python review/audit_w0_root.py /absolute/empty-evidence/cpu-ci-03 /absolute/root-audit.json
```

`agent-a/raw-manifest.json`, `agent-b/review-a-r1-raw-manifest.json`, `agent-c/raw-logs-manifest.json` and `review/scientific-positive-root-manifest.json` record per-file SHA-256 and byte counts. Their adjacent reports explain the restore directory. Original logs are not whitespace-normalized; expected-red outcomes, dependency failures and reviewer-harness errors remain labelled. No report treats plan-only output as an executed benchmark.

Source anchors use both original owner commits and published commits; Git-data publication verified exact tree equality. Scientific positive-control source b2ec565 and published source 71c7592 have identical full trees. Final delivery adds evidence/documentation to the tested product sources; the final PR HEAD also runs the same actual end-to-end CI.

The exact original local owner commits used by the independent probes are preserved in `source-anchors.bundle`, with refs/prerequisite/SHA-256 in `source-anchors-manifest.json`. From a clone containing baseline d66, import only the namespaced review refs:

```bash
git bundle verify evidence/runtime-next/source-anchors.bundle
git fetch evidence/runtime-next/source-anchors.bundle \
  'refs/astrolabe-runtime-20261104/*:refs/astrolabe-runtime-20261104/*'
```

This makes original SHA anchors such as 605bad0 and cf62f85 resolvable without moving any worktree branch. It contains committed source/review history only; existing untracked resource files are not included.


## f102 continuation (2026-11-04)

- [Recovered Runtime contracts](f102-runtime-host/README.md): real 309-test CPU process gate and complete archive.
- [Recovered Runtime W0 negative](f102-w0-reuse/README.md): no stable added performance gain; all 30 trials retained.
- [Native startup diagnosis](f102-startup-reuse/README.md): reused old work, real importtime and all negative controls.
- [Independent PyRosetta science](agent-c/f102-startup/README.md): actual installed native results, complete RNG/geometry and raw archive.
- [Independent startup W0](agent-c/f102-startup-w0/README.md): 30 valid trials, >20% CPU target, exact original raw records.
- [GPU checker resource review](gpu-control-resource-review/README.md) and [Slurm launcher review](gpu-launcher-resource-review/README.md): controlled CPU contracts, not GPU execution.
- [Root independent verification](root-f102-review/README.md): own comparisons, before/after counterexamples and preserved environment failures.
- `source-anchors-continuation.bundle` retains critical original source SHAs with a declared 9d prerequisite; `source-anchors-continuation.json` records exact refs/digest. The earlier source bundle remains unchanged.
- `decision.tsv` now includes baseline, patch, errors, work count, cost and decision columns. `decision-legacy.tsv` preserves the unmodified earlier log; unspecified old row fields explicitly point to its archived evidence.

Actual final-HEAD/GPU-admission results are attached on this separate evidence branch so
the tested source remains immutable. The ProteinMPNN preflight and launcher preparation
passed, but job 33334 never received an allocation: zero GPU forwards, zero completed
GPU trials, zero allocated GPU-hours. No pre-run matrix row is a real device certificate.
