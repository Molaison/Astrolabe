# Independent resource review of the pretrained GPU mutator control

Reviewer: Agent A. Date: 2026-11-04. Final reviewed source:
`2c6b26b96cbde76673b6cfa731dc6afda80ce2b5` (Agent B).

**Decision: PASS for the controller and offline evidence contracts reviewed here.**
This permits the resource owner to proceed with an independently authorized,
bounded Slurm allocation. This review launched no GPU job, model worker, or native
scientific evaluation, and does not certify actual CUDA execution or Slurm cleanup.

## Reproduced finding and repair

At `4889f7b7c08828a99b729719f0c668eaaf4d2396`, a JSON-roundtripped result could
contain `measurement.process_release.release_confirmed=false`, a nonempty remaining
PID list, and `observed_tree_empty=false`, while the duplicated row-level observation
still claimed successful release. The real `aggregate` returned PASS and published
certified effective throughput. The normal live controller creates both dictionaries
from the same observation; this was a saved-raw consistency defect, not an observed
GPU process leak.

Agent B repaired the checker in `2c6b26b`: the measurement release record must exist
and equal the duplicated observation in full; duplicated exit codes must agree;
observation errors, unverified groups, or stop reasons prevent certification.
Agent A reran **the exact original probe**: its unmodified positive fixture passes,
the contradiction becomes INVALID, and certified throughput is absent. Both complete
red and green fixture trees and stdout are in the raw archive. The checker additionally
requires a raw telemetry file before labeling telemetry COMPLETE; missing telemetry
continues to be explicitly unknown.

## Independent validation

| Gate | Source | Result | Scope |
|---|---|---|---|
| Original author suite independently rerun | `1b7a1b` | 52 passed, zero skipped, 8.40 s | CPU tensor fixtures |
| Final author suite independently rerun | `2c6b26b` | 69 passed, zero skipped, 18.71 s | CPU tensor fixtures; 69 is the final suite total, not added to the earlier 52 |
| Original raw-release contradiction | `4889f7b` | Expected failure: checker incorrectly PASS | Offline JSON evidence |
| Exact same contradiction after repair | `2c6b26b` | PASS: positive PASS, corrupted INVALID, no certified throughput | Offline JSON evidence |
| Independent controller boundaries | `4889f7b` | 8 of 8 passed | Synthetic run_once/clock/namespace/aggregate; real Popen prohibited |
| Controller/trial AST comparison | `4889f7b` to `2c6b26b` | Identical `run` and `trial` function ASTs | Carries the controller probes to the final checker-only revision |

The eight controller probes cover successful ordered execution of all ten trials;
nonzero child exit; unknown release; forced cleanup; a measurement exception; the
overall deadline; output budget exhausted before dispatch; and output budget
exhausted after one dispatch. Every failure stops subsequent trials. A pre-exhausted
budget launches zero trials. The positive case verifies list argv, exact allocated
UUID forwarding, 0.5 s observation cadence, decreasing remaining output allowance,
and observed-completion time distinct from the enclosing measurement duration.

The controller probe intentionally replaces the namespace predicate only while
`subprocess.Popen` is forbidden. It does not bypass an actual execution restriction.
Its small synthetic output budget is not a changed scientific workload or a measured
benchmark. Scientific work performed in this review: **zero**.

## Resource and caller contracts

The candidate compares the same revision's existing legacy native wrapper with its
existing private resident mutator process, with three sequential design requests per
model instance. It does not certify shared-broker concurrency, a pretrained evaluator,
single-trajectory MCMC, or a speedup between repository revisions.

The frozen protocol has ten trials in five interleaved groups, thirty native samples,
one GPU, four CPUs, 32 GiB memory, at most 900 s allocation time (0.25 GPU hours),
840 s inner experiment time, 150 s per trial, and 260,000,000 persistent output bytes.
There is no hidden warmup. The protocol SHA-256 remains
`72a18eef550fdf5555501b4163126522d752f441c3b7a5aad517a852d6498fc6`.

`run` and each `_trial` require Slurm allocation context and a single visible device,
reject inherited shared-broker credentials, and require a supplied actual GPU UUID.
The child compares that UUID with the CUDA device it really opened. Environment
strings alone do not prove scheduler ownership: root must establish a fresh allocation
and its UUID independently. The controller uses argv arrays without a shell; the
outer SIF invocation and its mounts remain the resource owner's responsibility.

The candidate reuses the unchanged `runtime_workflow_benchmark.run_once` and
`runtime_benchmark_processes.ProcessObserver`, preserving RSS/CPU and UUID-scoped GPU
telemetry, independent completion timing, PID/start-time release observations, and
failure cleanup. Their exact source hashes are recorded in `final-source.json` in
the archive. No second process observer or GPU scheduler was introduced.

The 260 MB limit is sampled persistent evidence/native output, not an atomic file
quota. Private temporary artifact exchange may live outside the trial directory;
the new allocation's TMPDIR/storage budget must cover it. Observed PID-tree release
is not a proof about an unobserved double fork; the external Slurm cgroup supplies the
final containment boundary. Full allocation startup, experiment, and shutdown time
must be recorded separately from the harness's inner experiment accounting. Missing
telemetry remains PARTIAL_UNKNOWN and is not converted into zero usage.

## Reproduction and raw evidence

Use a CPU-only environment with the already installed geometry dependencies. From
an independent checkout of the indicated B source, run:

```sh
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  /workspace/scratch/1a0b1639644d/cloud-20261104/geometry-venv/bin/python \
  -m pytest -q tests/test_gpu_mutator_control.py --junitxml=/NEW/PATH/tests.xml
```

The two adjacent probe scripts accept `REPOSITORY NEW_OUTPUT_DIRECTORY`. The
consistency script asserts rejection and therefore exits nonzero at `4889f7b`,
preserving the regression's real red result. Do not reuse an output directory.
The raw archive contains all 5,606 files from the review control directory, including
the earlier test fixtures, logs, JUnit, exact source pins, and independent probes.
`raw-manifest.json` gives every archived file's length and SHA-256 and the archive's
own hash. No synthetic output is presented as real pretrained GPU evidence.

One ancillary metadata command initially named the observer at the wrong path
(`astrolabe/runtime/process_observer.py`) and failed with FileNotFoundError. The
correct existing owner is `scripts/validation/runtime_benchmark_processes.py`;
`final-source.json` records the correction. This command did not modify source or
affect a test result.
