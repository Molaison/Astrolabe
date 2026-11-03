# Fixed-workload runtime comparison

Protocol SHA-256: `f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5`.

Complete fixed workload: **False**. Scientific score/decision gate: **False**.

| Mode | Valid / attempted trials | Completed trajectories | Median end-to-end seconds | Effective trajectories / second | Failed or incomplete trials | Allocated GPU hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 3 / 3 | 9 | 9.6880 | unavailable | 0 | 0.000000 |
| resident_serial | 2 / 3 | 7 | 32.9185 | unavailable | 1 | 0.000000 |
| resident_adaptive | 3 / 3 | 9 | 33.3308 | unavailable | 0 | 0.000000 |

## Interpretation

Each CLI run starts cold; per-backend/shape/phase cold and steady observations, loading counters, and control contexts are in runs[].runtime_diagnostics. No extra warm-up samples are generated.

Scores, selected sequences, stage and acceptance decisions must pass the repeated-direct baseline comparison before effective throughput is reported. Tolerances come from observed direct-repeat error; decision differences are checked separately. Coordinate and cache parity still require the backend's conformance evidence.

A ratio above 1 in the paired comparisons means the measured run finished faster. The runs include startup, all configured tasks, and shutdown. GPU activity is diagnostic and is not the throughput objective. CPU sampling may miss short-lived children; device activity and VRAM include other jobs unless allocation is exclusive.

Completion timing uses an independent process observer at 30 ms resolution. Allocated GPU cost also charges the trial's observation envelope; report.json records the entire comparison-loop allocation cost. Uncertain release makes resource costs observed lower bounds and stops subsequent trials.

## Paired end-to-end comparisons

| Repetition | Mode | Direct seconds / mode seconds |
| ---: | --- | ---: |
| 0 | resident_serial | unavailable |
| 0 | resident_adaptive | unavailable |
| 1 | resident_serial | unavailable |
| 1 | resident_adaptive | unavailable |
| 2 | resident_serial | unavailable |
| 2 | resident_adaptive | unavailable |

## Evidence

`plan.json` records the resolved scientific configuration, allocation, source identity, command and trial order. `report.json` retains all scientific rows, correctness differences, per-trial costs and runtime diagnostics. Each trial directory preserves stdout, stderr, raw telemetry and any runtime JSONL timeline. No profiler was enabled in timed runs.

Model load counters, cold/steady phase observations, resource peaks, release records and sampled low-activity contexts are retained under `runs[].runtime_diagnostics`. Missing counters stay unavailable. Composite native-call wall time is not labeled as device execution. A sampled control context does not prove an exclusive cause of GPU idleness.

## Correctness or completion limitation

```json
{
  "certified": false,
  "comparison": "scientific scores and decisions only",
  "direct_repetitions": 3,
  "coordinate_parity": "requires backend conformance evidence",
  "tolerance_source": "maximum same-cell absolute deviation from the first direct repeat",
  "tolerances": [
    {
      "identity": "(0, 17)",
      "row": 0,
      "field": "A.iptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 0,
      "field": "A.plddt",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 0,
      "field": "A.ptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 0,
      "field": "A_length",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 0,
      "field": "iptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 0,
      "field": "plddt",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 0,
      "field": "prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 0,
      "field": "ptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 0,
      "field": "weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "A.iptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "A.plddt",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "A.ptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "A_length",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "delta_iptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "delta_plddt",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "delta_prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "delta_ptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "delta_weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "iptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "mutator_selector_MCMC_pA_plddt_after_mean",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "mutator_selector_MCMC_pA_plddt_before_mean",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "mutator_selector_MCMC_pA_plddt_delta",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "mutator_selector_MCMC_pA_selected_count",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "mutator_selector_MCMC_plddt_after_mean",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "mutator_selector_MCMC_plddt_before_mean",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "mutator_selector_MCMC_plddt_delta",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "mutator_selector_MCMC_selected_count",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "plddt",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "previous_iptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "previous_plddt",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "previous_prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "previous_ptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "previous_weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "ptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 1,
      "field": "weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "A.iptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "A.plddt",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "A.ptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "A_length",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "delta_iptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "delta_plddt",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "delta_prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "delta_ptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "delta_weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "iptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "mutator_selector_MCMC_pA_plddt_after_mean",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "mutator_selector_MCMC_pA_plddt_before_mean",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "mutator_selector_MCMC_pA_plddt_delta",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "mutator_selector_MCMC_pA_selected_count",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "mutator_selector_MCMC_plddt_after_mean",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "mutator_selector_MCMC_plddt_before_mean",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "mutator_selector_MCMC_plddt_delta",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "mutator_selector_MCMC_selected_count",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "plddt",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "previous_iptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "previous_plddt",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "previous_prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "previous_ptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "previous_weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "ptm",
      "absolute": 0.0
    },
    {
      "identity": "(0, 17)",
      "row": 2,
      "field": "weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 0,
      "field": "A.iptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 0,
      "field": "A.plddt",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 0,
      "field": "A.ptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 0,
      "field": "A_length",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 0,
      "field": "iptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 0,
      "field": "plddt",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 0,
      "field": "prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 0,
      "field": "ptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 0,
      "field": "weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "A.iptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "A.plddt",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "A.ptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "A_length",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "delta_iptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "delta_plddt",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "delta_prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "delta_ptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "delta_weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "iptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "mutator_selector_MCMC_pA_plddt_after_mean",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "mutator_selector_MCMC_pA_plddt_before_mean",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "mutator_selector_MCMC_pA_plddt_delta",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "mutator_selector_MCMC_pA_selected_count",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "mutator_selector_MCMC_plddt_after_mean",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "mutator_selector_MCMC_plddt_before_mean",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "mutator_selector_MCMC_plddt_delta",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "mutator_selector_MCMC_selected_count",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "plddt",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "previous_iptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "previous_plddt",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "previous_prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "previous_ptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "previous_weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "ptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 1,
      "field": "weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "A.iptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "A.plddt",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "A.ptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "A_length",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "delta_iptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "delta_plddt",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "delta_prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "delta_ptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "delta_weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "iptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "mutator_selector_MCMC_pA_plddt_after_mean",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "mutator_selector_MCMC_pA_plddt_before_mean",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "mutator_selector_MCMC_pA_plddt_delta",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "mutator_selector_MCMC_pA_selected_count",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "mutator_selector_MCMC_plddt_after_mean",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "mutator_selector_MCMC_plddt_before_mean",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "mutator_selector_MCMC_plddt_delta",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "mutator_selector_MCMC_selected_count",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "plddt",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "previous_iptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "previous_plddt",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "previous_prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "previous_ptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "previous_weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "ptm",
      "absolute": 0.0
    },
    {
      "identity": "(1, 18)",
      "row": 2,
      "field": "weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 0,
      "field": "A.iptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 0,
      "field": "A.plddt",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 0,
      "field": "A.ptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 0,
      "field": "A_length",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 0,
      "field": "iptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 0,
      "field": "plddt",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 0,
      "field": "prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 0,
      "field": "ptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 0,
      "field": "weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "A.iptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "A.plddt",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "A.ptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "A_length",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "delta_iptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "delta_plddt",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "delta_prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "delta_ptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "delta_weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "iptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "mutator_selector_MCMC_pA_plddt_after_mean",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "mutator_selector_MCMC_pA_plddt_before_mean",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "mutator_selector_MCMC_pA_plddt_delta",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "mutator_selector_MCMC_pA_selected_count",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "mutator_selector_MCMC_plddt_after_mean",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "mutator_selector_MCMC_plddt_before_mean",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "mutator_selector_MCMC_plddt_delta",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "mutator_selector_MCMC_selected_count",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "plddt",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "previous_iptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "previous_plddt",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "previous_prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "previous_ptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "previous_weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "ptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 1,
      "field": "weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "A.iptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "A.plddt",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "A.ptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "A_length",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "delta_iptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "delta_plddt",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "delta_prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "delta_ptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "delta_weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "iptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "mutator_selector_MCMC_pA_plddt_after_mean",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "mutator_selector_MCMC_pA_plddt_before_mean",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "mutator_selector_MCMC_pA_plddt_delta",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "mutator_selector_MCMC_pA_selected_count",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "mutator_selector_MCMC_plddt_after_mean",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "mutator_selector_MCMC_plddt_before_mean",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "mutator_selector_MCMC_plddt_delta",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "mutator_selector_MCMC_selected_count",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "plddt",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "previous_iptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "previous_plddt",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "previous_prefix_steps",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "previous_ptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "previous_weighted_score",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "ptm",
      "absolute": 0.0
    },
    {
      "identity": "(2, 19)",
      "row": 2,
      "field": "weighted_score",
      "absolute": 0.0
    }
  ],
  "differences": [
    {
      "run": 9,
      "kind": "incomplete_workload"
    }
  ],
  "delivered_structure_parity": {
    "certified": null,
    "reason": "workflow emitted no structure artifacts; model coordinate parity not measured"
  }
}
```
