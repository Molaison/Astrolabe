# f102 recovered Runtime owner comparison — no additional speedup certified

**DONE (CPU correctness), negative/inconclusive performance result.** The recovered
resource lifecycle fixes and waiting-metric correction are retained for their independently
reproduced correctness effects. This experiment does not justify a throughput claim.

The unchanged W0 compares `9d62b3eb86cb75a2266af49dc48cb851e69e0341`
with `c90fadfdb82e2ba81b596b8d1cc93c4a25e783c4` on the same f102 SIF environment.
Candidate production matches the recovered runtime source `60684bd`; later GPU
control-checker edits are outside this CPU workflow. Neither absolute latency nor
improvement is compared across this SIF and the earlier GitHub runners.

Protocol SHA256: `f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5`.
Five interleaved A/B groups, three modes, 30/30 valid trials, 90/90 trajectories,
270/270 evaluations. All 90 original CSV files contain the expected 270 rows.
There are 90 accepted initialization rows and 180 rejected mutation attempts;
accepted mutation progress is zero. All observed process identities were released.
No structure artifacts or model coordinates were produced; no GPU was used.

| Mode | Baseline median/range seconds | Candidate median/range seconds | Aggregate valid eval/s gain | Faster pairs |
|---|---:|---:|---:|---:|
| direct | 10.7273 / 9.8941–15.7950 | 9.6860 / 9.3388–17.3680 | +4.2869% | 4/5 |
| resident serial | 33.2943 / 30.5781–41.3694 | 33.7630 / 31.2284–41.3118 | −0.1072% | 3/5 |
| resident adaptive | 33.2322 / 30.2160–37.8789 | 32.7041 / 32.2641–35.0838 | +0.8966% | 3/5 |

All ranges overlap. Aggregate rates use all scientifically valid evaluations and
all five measured workflow durations; no slow or failed work was discarded. This
supports retaining the correctness fix without claiming an added performance win.
The resident/direct startup regression remains substantial in this environment.

Full experiment wall time: **811.100408536 s** against the predeclared 900 s wall
budget and 260 MB output budget. GPU allocation: **0 hours**. Sampled CPU seconds
are lower bounds; wall time is not asserted to be total CPU consumption.

The archive is 254,596 bytes, SHA256
`1648312fcb0833d9821a863e9cf131f1813c7ecdbb9bd4e791f5658318818a7d`.
It contains all 634 original regular files (7,653,702 bytes), plus the per-file
manifest: frozen plan/config/environment, raw stdout/stderr, scores, provenance,
process identities/release evidence, JSONL telemetry/runtime traces, and original
report. Root downloaded and independently verified every file digest, recomputed
paired and aggregate timings, and reconciled raw CSV counts with process release.
`root-independent-audit.json` records the independent checks; `summary.json` keeps
the complete grouped statistics without duplicating the large raw trial records.

Exact execution (absolute root was the user-requested server directory):

```text
TASK_ROOT/control/test-container-cpu TASK_ROOT/candidate-c90fadf
  benchmarks/runtime/validate.py compare
  --baseline TASK_ROOT/repo-offline
  --baseline-commit 9d62b3eb86cb75a2266af49dc48cb851e69e0341
  --out TASK_ROOT/control/w0-reuse-c90-vs9d-01
```

The CPU wrapper explicitly hides all accelerators and fixes numerical thread
counts to one. The four original measurement modules remain the d66 bytes;
the wrapper checks baseline/candidate identity and the frozen workload contract.
