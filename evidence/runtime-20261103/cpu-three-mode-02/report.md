# Fixed-workload runtime comparison

Protocol SHA-256: `f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5`.

Complete fixed workload: **True**. Scientific score/decision gate: **True**.

| Mode | Valid / attempted trials | Completed trajectories | Median end-to-end seconds | Effective trajectories / second | Failed or incomplete trials | Allocated GPU hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 3 / 3 | 9 | 9.6914 | 0.273622 | 0 | 0.000000 |
| resident_serial | 3 / 3 | 9 | 32.4176 | 0.092256 | 0 | 0.000000 |
| resident_adaptive | 3 / 3 | 9 | 33.4545 | 0.090009 | 0 | 0.000000 |

## Interpretation

Each CLI run starts cold; per-backend/shape/phase cold and steady observations, loading counters, and control contexts are in runs[].runtime_diagnostics. No extra warm-up samples are generated.

Scores, selected sequences, stage and acceptance decisions must pass the repeated-direct baseline comparison before effective throughput is reported. Tolerances come from observed direct-repeat error; decision differences are checked separately. Coordinate and cache parity still require the backend's conformance evidence.

A ratio above 1 in the paired comparisons means the measured run finished faster. The runs include startup, all configured tasks, and shutdown. GPU activity is diagnostic and is not the throughput objective. CPU sampling may miss short-lived children; device activity and VRAM include other jobs unless allocation is exclusive.

Completion timing uses an independent process observer at 30 ms resolution. Allocated GPU cost also charges the trial's observation envelope; report.json records the entire comparison-loop allocation cost. Uncertain release makes resource costs observed lower bounds and stops subsequent trials.

## Paired end-to-end comparisons

| Repetition | Mode | Direct seconds / mode seconds |
| ---: | --- | ---: |
| 0 | resident_serial | 0.4309 |
| 0 | resident_adaptive | 0.4154 |
| 1 | resident_serial | 0.2910 |
| 1 | resident_adaptive | 0.2897 |
| 2 | resident_serial | 0.2900 |
| 2 | resident_adaptive | 0.2805 |

## Evidence

`plan.json` records the resolved scientific configuration, allocation, source identity, command and trial order. `report.json` retains all scientific rows, correctness differences, per-trial costs and runtime diagnostics. Each trial directory preserves stdout, stderr, raw telemetry and any runtime JSONL timeline. No profiler was enabled in timed runs.

Model load counters, cold/steady phase observations, resource peaks, release records and sampled low-activity contexts are retained under `runs[].runtime_diagnostics`. Missing counters stay unavailable. Composite native-call wall time is not labeled as device execution. A sampled control context does not prove an exclusive cause of GPU idleness.
