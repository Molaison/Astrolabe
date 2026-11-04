# Agent C evidence index

This directory records Agent C's producer correctness changes, independent reviews of Agents A/B, and a separate real CLI scientific positive control. The evidence commit changes no production code. Historical reports retain the state and scope at their original handoff; this index records the final disposition.

| Work | Final disposition | Entry point |
|---|---|---|
| Candidate-result identity pilot | **DONE**; root/B independently verified and root integrated | `pilot-report.md`, `pilot-manifest.json`, `pilot-identity-{before,after}.json` |
| Generation and duplicate submission guards | **DONE**; A/root independently verified; root integrated | `submission-report.md`, `submission-manifest.json`, `submission-{before,after}.json` |
| Bounded candidate queue | **PARTIAL**; naive blocking design rejected because it changes the selected candidate | `blocking-backpressure-negative.json`, `probe_blocking_backpressure.py` |
| A deadline/ownership review | **DONE** for scoped control-plane gates; two preexisting bugs found, fixed separately by A and independently retested | `review-a-report.md`, `review-a-*.json`, independent timeout probes |
| B ESMFold boundary review | **DONE** for scoped CPU/native-output contracts; pretrained GPU and workflow performance remain **BLOCKED** | `review-b-report.md`, `review-b-independent-differential.json`, `review-b-output-comparison.json` |
| Actual CLI accepted mutation and stage entry | **DONE** for the separate fixed CPU control; root also reports an independent integrated-HEAD replay | `scientific-positive-report.md`, `scientific-positive-02/report.json` |

## Negative evidence retained

The original foreign-result admission, mixed-generation and duplicate-submission failures are retained. The blocking backpressure counterexample retains the same work budget but a changed selection. A's before-fix deadline dispatch and delegation ownership failures remain available next to after-fix controls. B's original handler gate fails while the revised handler passes; the independent CPU output-only timings are slower by +9.31% and +6.68%, not a measured acceleration.

`scientific-positive-01` retains the mistaken five-backend-call expectation and **PARTIAL** report. Existing same-evaluator/same-oligo stage entry reuses a result and recomputes its score, so the corrected checker expects four backend calls and five score rows. `scientific-positive-02` uses the identical workload hash, seed and step budget and passes. All scientific outputs/RNG/owners also agree across those rounds; the lower count is no optimization claim. `scientific-control-evidence-checks.json` records the direct artifact-integrity checks.

## Raw log packaging

Raw `.log` files contain meaningful original output whitespace. They are preserved byte-for-byte in `raw-logs.tar.xz`, rather than edited to satisfy code whitespace checks. `raw-logs-manifest.json` lists the archive hash and each original relative path, size and SHA-256. All JSON/CSV/YAML scientific data, probes and reports remain directly present. The original loose logs also remain in the author's worktree, outside the commit.

To restore logs under a fresh evidence checkout, from this directory:

```sh
tar -xJf raw-logs.tar.xz
```

Archive members are relative paths confined to this directory. Verify them against the manifest when transferring evidence. No model weights, scientific structures, GPU results, secrets, process-transport certification or checkpoint-resume claim are included.

The frozen historical paths in plans/provenance identify where each run actually occurred. Replays use new output directories and a separately frozen source HEAD, with the same workload where a comparison is intended. They must not overwrite the original evidence.
