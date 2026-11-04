# Final HEAD independent CPU CI review

**PASS / DONE for this offline audit.** Candidate `9d62b3eb86cb75a2266af49dc48cb851e69e0341` versus baseline `d66ee714f2e755948534f95d9d125287ce8dbd91`. The original 20% goal remains unmet and resident remains slower than direct. No new benchmark, source modification or production commit was made.

## Raw evidence and scientific work

All 645 raw artifact files were hashed before and after analysis. All 30 trials, 90 original CSV/provenance/config sets, 30 trial configs, runtime traces and telemetry streams were checked; the author aggregate summaries and embedded scientific rows were not used. The audit recomputes performance from individual persisted observer measurements.

| Check | Observed result |
|---|---:|
| Completed trials / trajectories / evaluator calls | 30 / 90 / 270 |
| Mutation attempts / accepted mutations | 180 / 0 |
| Initial rows / accepted initial rows | 90 / 90 |
| Stage entries / accepted improving mutations | 0 / 0 |
| Process releases / empty terminal broker ledgers | 30 / 20 |
| Scientific rows and raw CSV bytes, corresponding repeat/seed | Exact across all 30 trials |
| Full trial / executed run configurations verified | 30 / 90 |
| Exact Git source-file hashes verified | 433 baseline + 436 candidate |

Every accepted row is initialization. W0 therefore provides completion-throughput and rejected-proposal parity, not evidence of accepted scientific progress. The independent accepted-mutation/stage-entry control remains separate. No GPU, pretrained model, structure, full RNG snapshot, checkpoint or retry recovery is certified by W0.

## Final-run performance

| Mode | Baseline median seconds [range] | Candidate median seconds [range] | Median paired throughput gain | Median paired elapsed reduction | Faster pairs |
|---|---:|---:|---:|---:|---:|
| direct | 2.9347 [2.8565–3.2777] | 2.6723 [2.6415–2.7453] | +9.48% | 8.66% | 5/5 |
| resident_serial | 7.5480 [7.4678–7.6758] | 6.7115 [6.5728–6.7799] | +13.62% | 11.99% | 5/5 |
| resident_adaptive | 7.6069 [7.4841–7.7559] | 6.6627 [6.6133–6.8662] | +13.17% | 11.64% | 5/5 |

Each same-mode pair improves and each candidate range lies strictly below its baseline range. These are five interleaved cold-start groups on one runner, not population confidence bounds. Throughput gain is baseline/candidate elapsed ratio minus one; elapsed reduction uses its reciprocal. The individual ratios are retained in `report.json`.

Candidate resident serial/adaptive throughput versus the best direct revision has paired median ratios **0.401882× / 0.399836×**. Resident elapsed times are **2.4883× / 2.5010×** direct, with **0/5** resident-faster pairs in each mode. Resident does not reach direct parity or the original 20% throughput goal. All three same-mode paired median gains are also below 20%.

Absolute times differ from earlier CI runs. The runner identities are recorded in `source-equivalence.json`; no samples or absolute times from different CI runs were pooled. The correct final-HEAD medians are those above, not the earlier 71c7592 medians.

## Frozen sources and measurement integrity

Complete configuration, protocol hash, four measurement-file hashes and budgets agree with prior CI 01, 02 and 03. The final candidate running-source manifest is byte-identical to candidate 71c7592. The frozen protocol remains `f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5`. All source hashes match the actual Git objects at the expected final revisions. Mode order, alternating AB/BA revision order, exact repeat/seed set, nonoverlapping trials and empty GPU allocation were verified.

The process-completion endpoint and origins reconstructed from raw telemetry reproduce the persisted elapsed values within **1.924 microseconds**. This consistency check does not increase the original observer's approximately 30 ms completion resolution or provide a new timing experiment.

## Reproduction and outputs

The unchanged existing reader is `evidence/runtime-next/agent-c/review_r1_ci.py`, SHA-256 `eda968172261e7b266278a4616c65d17b8b70582f3e670b41b7249dff45f5a3a`. Its earlier fixed candidate expectation was changed only in the analysis process to the explicitly requested final revision; no code file was edited. `invocation.json` records the expected revisions, exact input directories, prior plans, reader hash and successful exit.

A portable wrapper is provided as `reproduce.py` next to this report. It verifies the original reader SHA-256 before loading it, applies only the recorded in-memory final-HEAD expectation, and calls the same unchanged audit. The clone must contain the baseline/final Git objects and the committed reader. The Python environment needs OmegaConf (the project CPU validation environment already supplies it). The raw artifact directories are the extracted bundle data; no network or benchmark execution is involved.

From a bundle/workspace root containing `repo/` and `control/`, the exact command is:

```sh
python control/final-ci-review/reproduce.py \
  --repo repo \
  --artifacts control/cpu-ci-04 \
  --prior control/cpu-ci-01 \
  --prior control/cpu-ci-02 \
  --prior control/cpu-ci-03 \
  --out control/NEW-final-ci-review
```

For a fresh clone/bundle in other locations, change only those path arguments; none embeds the original workspace path. The output directory must not already exist. The wrapper was syntax-checked; the completed audit above used the equivalent recorded in-memory invocation and was not rerun during packaging.

Use a new output directory for a replay. `report.json` retains counts, errors, full trial measurements, exact CSV hashes, paired comparisons and limits. `input-manifest.json` inventories all 645 raw files; its SHA-256 is `e3bc28436779d7c0e010ecc401ab6c18ccbccd42eb847fe8a23a8e545fbbac29`. The coordinator can publish these read-only results on the final-evidence branch without changing the verified PR HEAD.
