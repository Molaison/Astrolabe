# Independent final HEAD validation — Agent C

## Packaged snapshot

`final-head-c-raw-and-review.tar.xz` preserves all 710 evidence files plus their
snapshot manifest, including both original GitHub ZIPs and all extracted raw.
It is 978288 bytes, SHA-256 `38adc3042896c0721787476fec737141dbe851ddf6ee1ffbc90f9f596de6b18a`. Every archived file was independently rehashed
against the source snapshot after packaging. `RAW_MANIFEST.json` lists each
file, byte length and SHA; it does not count itself.

From this package directory, verify the archive and extract into a fresh folder:

```bash
printf '%s  %s\n' 38adc3042896c0721787476fec737141dbe851ddf6ee1ffbc90f9f596de6b18a final-head-c-raw-and-review.tar.xz | sha256sum -c -
mkdir extracted
tar -xJf final-head-c-raw-and-review.tar.xz -C extracted
cd extracted/final-head-c
```

The following report and reproduction commands refer to that extracted layout.
The top-level scripts and summary are exact reviewable copies of their archived
counterparts. This is a data-only evidence branch based on the tested production
HEAD, keeping the published source unchanged.

**DONE for the exact final CPU CI and direct scientific positive-control scope.**
The published source is `1bdaf3a53227c9197b930dc00a01b752acd90067`, tree
`6f763bfb04778ef4f32386ddb8aeb38ce4240bf1`. Agent C created a detached checkout;
it was clean before and after the actual positive CLI control. Its production
`astrolabe/`, `run.py` and `config/` bytes are identical to the earlier tested f9
startup candidate. No production file, workload, measurement or budget changed.

## Current CI identity and complete artifacts

The exact `head_sha` query found Runtime CPU validation
[run 41780311942](https://github.com/Molaison/Astrolabe/actions/runs/41780311942),
attempt 1, push, completed/success. Every step in both jobs succeeded. This is
the newly triggered final-source run, not historical run 41766956074. The same
HEAD's separate release push 41780311999 and PR run 41780318835 were also reported
successful; this archive preserves all artifacts of the Runtime workflow only.

| Job | ID | Artifact ID | Raw files | Independent JUnit result |
| --- | ---: | ---: | ---: | --- |
| runtime-cpu | 126410101922 | 13299246383 | 645 | 378 passed, 0 failed, 0 skipped |
| CPU Torch and structural contracts | 126410102337 | 13298249397 | 15 | 181 passed, 0 failed, 0 skipped |

Both original ZIPs were downloaded completely. Their lengths and SHA-256 match
the GitHub artifact API; every extracted file has a SHA/byte-length inventory.
The original ZIP bytes are retained alongside their complete extractions.

- CPU ZIP: 1,451,269 bytes; `d7d70c6557c66a336bdc4dd63ed3d55b03a063ea87ca2df4146266adf9a1b002`.
- Torch ZIP: 10,953 bytes; `cd87c4af4c2e3c1ed931b4fa3c8f888329081a727e6889101bfa7c0a3ae5e1ee`.

`api/` preserves the exact run, jobs and artifacts API responses. The independent
gate reader parses the final Git object's REQUIRED_TESTS/TORCH_CONTRACT_TESTS,
checks the actual pytest argv, checks every JUnit testcase and suite, and rejects
missing, duplicate, failed, skipped or uncollected mandatory tests. All 28 core
and 11 geometry/Torch suites have actual testcases. The PyRosetta lazy suite is
mandatory in geometry/Torch, where RDKit is installed; it was not skipped or
added to the frozen minimal CPU dependency set. Torch is 2.7.1+cpu with no CUDA
build. No pretrained/GPU inference follows from these contract tests.

## Independent final W0 raw audit

The fixed original reader SHA is
`eda968172261e7b266278a4616c65d17b8b70582f3e670b41b7249dff45f5a3a`.
`review_final_ci.py` overrides only the expected candidate HEAD in memory;
its original source, scientific and timing/release logic are unchanged.
It reads all original CSVs, full configs, provenance, runtime traces and
individual observer records, not the author's aggregate conclusions.

**30 trials, 90 completed trajectories and 270 actual evaluator calls; zero
audit errors.** All complete scientific rows match exactly by repeat/seed.
All source manifests match the exact baseline d66 and final Git objects.
All 20 terminal resident ledgers are empty and all observed process trees are
released. Full workload, measurement hashes and budgets match prior CI01–04;
the ten frozen fields also match the separate f102 f9 plan. No new W0 run was
started by Agent C.

| Mode | d66 median [range], seconds | Final median [range], seconds | Median paired throughput gain | Aggregate throughput gain |
| --- | ---: | ---: | ---: | ---: |
| direct | 2.5164 [2.4847, 2.9207] | 1.5842 [1.5490, 1.6282] | +60.4016% | +63.8104% |
| resident_serial | 6.5937 [6.5217, 6.7618] | 3.6120 [3.5157, 3.6505] | +81.9017% | +84.3350% |
| resident_adaptive | 6.6293 [6.5286, 6.8108] | 3.5588 [3.5187, 3.6121] | +87.4674% | +86.6939% |

Every mode has 5/5 faster pairs with disjoint before/after timing ranges. The
same-mode 20% CPU engineering threshold is met. Aggregate throughput is equal
completed work divided by summed observed time; paired gains are the median of
baseline-time/candidate-time ratios. They are distinct statistics. This is five
cold interleaved pairs per mode on one CI runner, not a population interval or
long steady-state break-even estimate.

Candidate resident serial/adaptive still take 2.2981/2.2464 times the candidate
direct time on the paired median, with 0/5 wins over direct. The native f102
f9-versus-9d experiment is a separate environment and baseline; its absolute
times and gains are not pooled with this CI result.

All 90 accepted W0 rows are initial evaluations. All 180 mutation attempts are
rejected, with zero accepted mutation and zero stage entry. Thus W0 certifies
equal fixed-work completion and recorded science, not accepted-mutation progress,
pretrained inference, GPU throughput, structures, full RNG or crash-resume.
Process release is limited to the observer's recorded scope.

## Actual final-source scientific positive control

The unchanged committed `runtime_scientific_control.py` froze a fresh plan before
execution, then invoked the real public CLI once per source: d66 versus final.
Each source uses direct CPU Dummy/MCMC, seed 17, one repeat, four steps, no warmup,
at most 45 seconds and 16 MiB output. The fixed two-stage workload is independent
of W0 and was not retuned. Workload SHA:
`ddbd295188ed22241736750bed4fa613c553dbf29b5423de0f9454b7c90cbde9`;
harness SHA:
`6f52867a3d45e83985fc84b844cfdf7227e61761e615355a63b913d83eb4ed26`.

Both runs completed: **4 actual evaluator calls, 5 score rows, 3 mutation
attempts, 1 accepted sequence mutation at step 1 and 1 stage entry**. The extra
score row recomputes the stage score on the existing cloned result; it is not an
extra model call or an optimization introduced in this run. The earlier
expected-five-calls negative result remains in prior evidence.

The entire original CSV, terminal scientific owner, Python/NumPy/Torch CPU RNG
states and all scientific output hashes match exactly. NumPy records its full
624-key state and Torch its full 5,056-byte CPU state. The observer does not
consume RNG, reseed or patch any scientific function. The separate independent
gate reader rechecks the raw CSV, runtime phase counts and terminal snapshots,
not merely the harness report. Local interpreter was Python 3.12.14 with the
existing project venv; both sides used the same environment. These data are
correctness-only and are not mixed into any throughput denominator.

This control has no structures, native Rosetta loss, pretrained model or GPU;
terminal-state equality is not checkpoint/resume certification. The separate
actual SIF PyRosetta first-use control remains separately scoped evidence.

## Reproduce from the preserved data without new computation

After extracting this directory, use a Git repository containing d66, final
1bdaf3a and the delivered f9 source object. `$REPO` points to that repository.
The independent W0 reader needs OmegaConf; the JUnit/positive raw reader only
uses the standard library. Output paths must be fresh.

```bash
python review_final_ci.py --reader review_r1_ci.py --repo "$REPO" --artifacts cpu-artifact --prior prior-plans/ci-01 --prior prior-plans/ci-02 --prior prior-plans/ci-03 --prior prior-plans/ci-04 --out ../reproduced-w0
python review_final_gates.py --root . --repo "$REPO" --output ../reproduced-gates.json
```

Original absolute paths in plan/command/observer JSON are retained as provenance;
the readers resolve persisted data by their extracted relative layout. They do
not invoke any source workload. The optional new actual CLI rerun is separate:

```bash
python "$REPO/scripts/validation/runtime_scientific_control.py" freeze --baseline "$BASELINE_REPO" --candidate "$REPO" --output new-scientific-control
python "$REPO/scripts/validation/runtime_scientific_control.py" run --plan new-scientific-control/plan.json
```

`$BASELINE_REPO` must be a clean detached d66 checkout; `$REPO` must be the final
clean source for that actual rerun. The harness refuses changed science/config,
workload, budget or measured-reader identity. This evidence submission contains
no product changes and makes no real pretrained GPU or full-backlog DONE claim.
