# Independent Slurm launcher resource review

Agent A, 2026-11-04. Final reviewed source:
`959226413c02ff85989f002b9fb9a4100849269a` (Agent B).

**DONE: independent code and CPU-boundary review.** The launcher may proceed to
root's actual CPU preparation gate and, only after that gate, root's bounded new
allocation. **PARTIAL: actual host compatibility, scheduler admission, assigned
GPU execution, and terminal allocation release remain to be observed.** No command
in this review submitted a job, queried a GPU, started a container, or used CUDA.

The review began at `ed8eeeb76c82e355e0cc922498ab11f993488f0f`, after root's earlier
ordinal-remapping finding had already been repaired. A did not count that prior
fix as a new A finding. The scientific harness and frozen protocol hashes are
identical across this review and its repair.

## Reproduced findings and repair

| Finding at ed8 | Reproduced outcome | Narrow repair in 959 |
|---|---|---|
| Supported complete UUID invalidated between callers | `allocated_binding` normalized `GPU-...` to `gpu-...`; `gpu_query_argv` rejected its own caller's token | Keep the canonical `GPU-` prefix and normalize the hex portion; the entire supported call chain now succeeds |
| Inherited display format survives UTC normalization | An inherited `SLURM_TIME_FORMAT=relative` remained active, although the clock reader expects complete dates | Both the new srun environment and the outer sacct query explicitly use UTC and standard date formatting |
| Contradictory terminal evidence could be labeled within budget | Non-dates, end-before-start, and aggregate GPU count 1 with typed count 2 were accepted | Parse complete UTC dates, require ordered endpoints, validate exit-code form, and require all present GPU counts/types to agree with one g4090 |

These were supported-input and evidence-acceptance defects reproduced with
synthetic scheduler records. They are not observations of actual GPU leakage or
incorrect Slurm accounting. The complete original red script and records are
preserved. The exact same independent probe changes from **2 of 7 expected
outcomes passing to 7 of 7 passing** on the repaired source.

The repair keeps both scheduler ElapsedRaw and the UTC endpoint span; it does not
assume those fields are identical. Either exceeding 900 s marks a budget failure.
Missing or delayed terminal accounting stays PARTIAL rather than implying release.

## Independent validations

| Validation | Source | Result |
|---|---|---|
| Author tests independently rerun | ed8 | 42 passed, zero skipped |
| Author tests independently rerun | 959 | 50 passed, zero skipped |
| A binding/accounting adversarial probe | ed8 → 959 | Preserved red → all 7 outcomes pass |
| A caller-level lifecycle probe | ed8 and 959 | All 5 cases pass at both revisions |

The five caller-level cases execute the real `submit`, `inside`, or termination
logic around explicitly controlled child objects and metadata providers:

1. A successful submission has exactly one new bounded srun argv; repeating the
   submission creates no second Popen and leaves the original record unchanged.
2. Delayed accounting returns `PARTIAL_ACCOUNTING_OR_BUDGET`, without a fabricated
   terminal record.
3. A timed-out child receives terminate, bounded wait, kill, bounded wait through
   its own Popen object. A separate unrelated child records no actions.
4. A foreign UID in the inside caller's scheduler response fails before any GPU
   query or container launch; all before/after queries remain scoped to the same
   indicated job/step.
5. An already exited child is only polled and receives no signal.

The Popen/command boundaries in these probes are controlled test doubles. They
test decisions and ownership scope, not real process reaping, scheduler behavior,
native GPU execution, or performance. No namespace safeguard is bypassed for real
execution. Full logs, JUnit, per-case records, and scripts are archived.

## Resource contract assessed

The launcher prepares its source, frozen assets, full existing SIF checksum and
CPU environment before requesting a GPU. It creates a new private evidence/temp
directory within cloud-20261104, preserves failures, and does not overwrite an old
worktree. Readonly source/assets and private temporary/cache binds remain explicit.

The only allocation argv is a fresh one-node, one-task job in xulab / normal /
boron, one g4090, four CPUs, 32 GiB, 900 s and a 60 s admission wait. There is no
attachment to an existing job, retry, node pin, node exclusivity, or shared-GPU
selection. The inside caller verifies the unique job name, UID, job/step identity,
allocation sizes/type/state and clock before launching the scientific harness.

The scheduler's assigned global ID is the sole target of the metadata query. Its
resolved UUID is then the sole container CUDA/NVIDIA visible device, including
the global-index-5/inherited-ordinal-0 case. The inherited ordinal is retained as
provenance. Device visibility alone is not scheduler ownership or containment;
actual Slurm enforcement and the later native UUID check remain necessary.

Termination targets the launcher-created child object. A live step's completion
is not treated as terminal allocation release; the outer command obtains only its
recorded new job's sacct row. Actual allocation endpoints and elapsed time remain
separate from client queue time and the scientific harness timer. A real delayed
accounting result may require a later readonly check, without rerunning science.

## Primary semantics checked

Official [Slurm srun documentation](https://slurm.schedmd.com/srun.html) identifies
the job/step GPU environment fields as global IDs, not cgroup-relative ordinals,
and documents srun signal forwarding. This supports the global-assignment query,
UUID binding, and owned-srun termination model; it is not a replacement for an
actual allocation and release record.

Official [scontrol documentation](https://slurm.schedmd.com/scontrol.html) and
[sacct documentation](https://slurm.schedmd.com/sacct.html) describe the
`SLURM_TIME_FORMAT` override. The latter defines the elapsed-seconds field and
terminal Start/End fields separately. The review therefore tests explicit date
formatting and parseable ordered endpoints while retaining both time measures.
These primary pages were consulted during this review; no third-party scheduler
assumption is used as evidence of f102 availability.

## Reproduction and integrity

Using the existing geometry venv in an independent checkout of each indicated
source, run `python -m pytest -q tests/test_gpu_slurm_launcher.py`. The adjacent
`probe_binding_accounting.py` takes `REPOSITORY NEW_RESULT_JSON`; its assertion
fails at ed8 and passes at 959. `probe_controller_lifecycle.py` takes
`REPOSITORY NEW_OUTPUT_DIRECTORY` and preserves each complete caller record.

`raw.tar.xz` is a lossless copy of the local review directory. Its per-file lengths,
SHA-256 values, and archive SHA are in `raw-manifest.json`. Source identity and
unchanged harness/protocol hashes are in `final-source-and-validation.json` inside
the archive. The next actual CPU/Slurm/GPU gates belong to root; the code-review
PASS above must not be substituted for those observations.
