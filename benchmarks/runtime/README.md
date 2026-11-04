# Fixed CPU runtime validation

`validate.py` wraps the existing `scripts/validation/runtime_workflow_benchmark.py`.
Its `run_once`, scientific comparator, structure comparator and process observer
remain unchanged. This is CPU Dummy protocol evidence, not a model or GPU speed
test. The original release workflow is unchanged.

The scientific projection must match SHA-256
`f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5`:
AAAA, MCMC random proposals/positions, Dummy, sequence-residue fraction,
three repeats with seeds 17/18/19 and three steps per repeat.

The separate baseline checkout must match the exact requested commit, with
clean runnable source. The default remains
`d66ee714f2e755948534f95d9d125287ce8dbd91`. An explicit
`--baseline-commit 9d62b3eb86cb75a2266af49dc48cb851e69e0341` compares a later
candidate against the independently validated CPU revision, so that a regression
relative to that revision cannot be hidden by the slower original baseline.
Only these two pinned commits are accepted. The four measurement modules still
must match their immutable **d66 SHA-256 values** in the baseline, candidate and
wrapper checkouts, even when both compared revisions contain the same drift.
Plans, comparison reports and summaries record the actual baseline HEAD; a
missing source identity is reported as unavailable. Both
revisions use the same Python 3.11 environment and the same composed scientific
config. Five groups rotate direct/resident-serial/resident-adaptive mode order;
each mode runs both revisions, alternating AB/BA by group. There are **30 trials,
90 requested trajectories and 270 requested evaluations**, no extra warmup,
with a **900 s total comparison wall budget and 260,000,000 byte output budget**.
The previous 15-trial plan remains historical evidence, not this comparison.

Every completed trial requires exact trajectory identities, three score rows
(steps 0/1/2) per trajectory, and nine recorded Dummy prepare/execute/finalize
calls. A missing trace is unavailable evidence, never zero evaluations.
All score, sequence, acceptance and stage fields pass the existing scientific
comparison across both revisions and all modes. Only the five **baseline direct**
runs define numeric tolerance; candidate direct results cannot increase it.
Verified process-tree release is required. Incomplete work, scientific failures
or missing resource evidence suppress all speed ratios and effective throughput.

The report retains individual elapsed times, medians/ranges, same-mode pairs,
and candidate ratios to the direct revision with the lower median. Five paired
improvements and nonoverlapping ranges are shown separately; the wrapper does
not call a noisy difference a proven improvement. No structure coordinates are
emitted by W0, so coordinate/model/cache certification remains unavailable.

GitHub Actions installs `requirements-cpu.txt` without model packages or large
submodules, validates Linux process namespace/UNIX sockets, then runs every
required CPU suite without accepting skips. Missing pilot tests are blocking.
The recovered Runtime ingress, request-isolation and saturation suites are also
required; a checkout missing any of them is **BLOCKED**, not silently skipped.
JUnit includes any skip reasons. Framework/GPU-dependent suites not listed in
the gate are outside its certification scope, not implicitly passing tests.

A separate `runtime-torch-cpu` job installs the same base dependencies,
`torch==2.7.1+cpu` from the official CPU wheel index, and pinned Pandas/RDKit
from `requirements-geometry.txt`. It runs the four fixed
output-boundary, tree-codec, RNG-lifecycle and ESMFold1-resident suites with the
same process/IPC preflight and zero-skip JUnit gate (`--torch-contracts`), plus
real SciPy/RDKit structural loss, Qhull fallback and lazy-import/RNG controls. Its
required suites also include `test_gpu_mutator_control.py`, which tests the GPU
control harness's scientific and failure gates using CPU tensors. These checker
tests are mandatory after integration; they do not certify a GPU model run. Its
runner, logs and artifact are separate from W0; Torch is never added to the W0
environment; RDKit and Pandas also remain outside W0. These are CPU contract tests, including local fake-weight model
fixtures, and do not certify pretrained inference or GPU acceleration.

For local planning only (safe on an executor without process ownership):

```bash
python benchmarks/runtime/validate.py compare --baseline /absolute/baseline-checkout --out /absolute/new-plan --plan-only
```

For the additional regression comparison against the validated 9d revision:

```bash
python benchmarks/runtime/validate.py compare \
  --baseline /absolute/9d-checkout \
  --baseline-commit 9d62b3eb86cb75a2266af49dc48cb851e69e0341 \
  --out /absolute/new-9d-plan --plan-only
```

Remove `--plan-only` only in the required execution environment to run that
comparison. The selected commit changes neither W0 nor the five-group,
30-trial budget, mode order, scientific gates or measurement implementation.
The existing CI command omits the option and continues to compare against d66.

For execution, use the workflow environment (CPU visibility, one BLAS/OpenMP
thread, Python 3.11). Each command requires an empty evidence subdirectory:

```bash
python benchmarks/runtime/validate.py preflight --out /absolute/evidence/preflight
python benchmarks/runtime/validate.py regressions --out /absolute/evidence/regressions
python benchmarks/runtime/validate.py compare --baseline /absolute/baseline-checkout --out /absolute/evidence/comparison
python benchmarks/runtime/validate.py summary --out /absolute/evidence
```

The workflow uploads raw stdout/stderr, config files, scientific rows and
provenance, JSONL traces/telemetry, source identities, test results and summaries
even on failure. It also writes the substantive result into `GITHUB_STEP_SUMMARY`.
No denied process/IPC capability is bypassed and no shared GPU is discovered or
allocated. Failure to obtain an actual CI run is **BLOCKED**, not a benchmark.
