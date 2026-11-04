# Fixed CPU runtime validation

`validate.py` wraps the existing `scripts/validation/runtime_workflow_benchmark.py`.
Its `run_once`, scientific comparator, structure comparator and process observer
remain unchanged. This is CPU Dummy protocol evidence, not a model or GPU speed
test. The original release workflow is unchanged.

The scientific projection must match SHA-256
`f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5`:
AAAA, MCMC random proposals/positions, Dummy, sequence-residue fraction,
three repeats with seeds 17/18/19 and three steps per repeat.

The separate baseline checkout must be exactly commit
`d66ee714f2e755948534f95d9d125287ce8dbd91`, with clean runnable source. Both
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
JUnit includes any skip reasons. Framework/GPU-dependent suites not listed in
the gate are outside its certification scope, not implicitly passing tests.

For local planning only (safe on an executor without process ownership):

```bash
python benchmarks/runtime/validate.py compare --baseline /absolute/baseline-checkout --out /absolute/new-plan --plan-only
```

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
