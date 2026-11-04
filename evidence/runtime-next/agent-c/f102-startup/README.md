# Independent scientific review of recovered native startup optimization

## Current status

**Source review PASS; installed native execution and local raw reanalysis PASS.** This
directory belongs to Agent C. Production files are read-only. The candidate is
`f9ce87b9f5cf37710f0a52ebe503595e5d7fdd7f`; the fixed source baseline is
`47c6f6dd18fe822a6d28ea6970c756d5dfae7b95`.

The root confirmed that the formal `c90` versus `9d` W0 comparison ended.
Agent A completed its four bounded cold-start diagnostic processes before Agent
C ran the two native correctness processes. The native control below is not an
end-to-end performance measurement or pretrained GPU certification. The separate
section at the end independently audits the already completed W0 raw data.

## Actual native outcome

Both exact clean sources completed the real PyRosetta control with exit code 0,
released process groups, and CUDA uninitialized. The installed native version was
PyRosetta 2026.03+releasequarterly.5e498f1409c68ade56c8ce5842bf79e1b02e8db4,
Python 3.11 and Torch 2.7.1+cu126. All 169 registered losses retained their order,
full metadata and callable signatures; all 169 per-loss runtime requirements and
the 40-name PyRosetta classification set matched exactly. All seven imported
native symbols had the required actual module/class identity.

Python, NumPy and Torch CPU states matched exactly after the production seed,
registry/requirements query, first native require, explicit native init, native
pose construction and loss computation. The first require did not initialize
Rosetta or consume those RNG streams. Neither scientific state restoration nor
post-import reseeding was used to make this pass. The diagnostic cold-import
states also matched.

The actual 27-value native SASA arrays sum to **2399.656935207131 Å²**. Both
sources returned `chain_buried_fraction = 0.28873283581553477` and
`com_distance = 9.000000000000002 Å` on both calls. Full arrays, every atom name
and coordinate, repeated-call results and unchanged input geometry are exact.
The candidate cold import defers both PyRosetta and scipy.spatial.

The two wrapper/native processes consumed **30.020256 CPU seconds total**, below
the 90-second control budget; wall times were approximately 14.98 and 14.43 s.
Those times are recorded only for budget accountability, without a performance
claim. No scientific process overlapped A's profiling or the previous formal W0.

All 18 native raw files were retrieved and individually SHA-256 verified. Their
1,099,987 original bytes are preserved in `native-raw.tar.xz` (37,372 bytes), SHA
`c5a4b6f2933322542e04aa10c862e1524071958bb4b28f82ea613f46b8fe7d5c`.
`native-comparison-local.json` is the independent rerun of the exact comparator
on those downloaded raw files. It passes all 28 recorded comparison gates.

## Caller and preserved science

The production path imports the runner before `argparser.set_random_seeds`.
The baseline eagerly imports PyRosetta during loss discovery; the candidate
imports it when an existing native loss/helper first calls `_require_pyrosetta`.
This change can move dependency import side effects across the scientific seed
boundary. Consequently, the native control snapshots the actual Python, NumPy,
and Torch CPU states before and after first use; it never restores those states
to make a comparison pass.

`source-audit.json` comes from immutable Git objects, not the moving integration
worktree. It independently confirms 136 unchanged top-level interaction functions,
six helpers with only an added leading guard, and `_require_pyrosetta` as the sole
other changed function. The signatures/decorators of all 134 direct registered
function definitions remain identical. Dynamic aliases are a separate installed
registry check, so the static count is not presented as the full registry count.
The earlier read of the moving integration tree was retained separately as
`source-audit-working-tree-observation.json`; it is not the frozen comparison.

The six guards cover chain selectors, per-residue SASA helper selection, holes,
pose loading, and all-alanine mutation. The existing scientific function bodies,
initialization callers, acceptance logic, RNG ownership, and stage transitions
are unchanged. Source search found no production caller importing a frozen copy
of the old `interaction.pyrosetta` global. The previously verified SciPy lazy
imports are preserved.

## Frozen native control

`native-control.json` fixes the complete control before native execution. It
reuses the full `ACDEFGHIK` pose fixture from the existing PyRosetta tests and
builds three deterministic translated copies, without cropping residues or atoms.
The 27-residue ABC pose runs the actual `PerResidueSasaMetric` through
`sasa_calculator`, plus registered `chain_buried_fraction` and `com_distance`
losses twice. The geometry before/after, full 27-value SASA arrays, and all
registered loss outputs are retained. This is a separate correctness input,
not a reduced form of W0 or the GPU ProteinMPNN control.

Each process records every installed registry name in order, full dataclass
metadata, callable module/qualname/signature, every registered loss's runtime
requirements, and the complete PyRosetta classification set. The seven loader
symbols must be the real imported module/classes; lazy import must not call
`pyrosetta.init`. Native initialization is explicit with identical frozen flags
and the Rosetta seed 17 on both revisions. The production seed helper establishes
Python/NumPy/Torch seed 17 once after cold runner import, with no later reseeding.

Native gates require exact metadata, requirements, RNG states after each
scientific phase, coordinates, and numeric outputs. Differences are failures;
there is no post-hoc tolerance relaxation. CUDA must remain uninitialized.
The control directly validates two registered native losses and symbol loading;
it does not certify every PyRosetta metric, DAlphaBall, or GPU backend.

## Budget and execution

Remote stage:
`/xcf26/zpzeng/practice_arena/astrolabe_runtime_20261102_implementation/cloud-20261104/control/c-startup-science-01`

Root-provided clean clones:

- `TASK_ROOT/startup-baseline-47c6f6d`
- `TASK_ROOT/startup-candidate-f9ce87b`

The root-owned `TASK_ROOT/control/test-container-cpu` wrapper was independently
read: it uses the actual licensed SIF, hides all accelerator device variables,
and sets OMP/MKL/OpenBLAS/NumExpr/VecLib thread counts to one. Its SHA-256 is
`6c295bc7fb9edb2c8ce0fa6c04517df2e3fcfd07444c3cf968a167c7d310ad01`.

`run_native_pair.py` creates an exclusive execution directory and invokes only
the two protocol-fixed processes sequentially. Each native process has a
45-second soft CPU limit and 85-second wall limit. Before the second process the
controller reserves its full CPU budget; total observed host/controller/child
CPU above 90 seconds fails. A timeout, nonzero exit, incomplete native record,
or unreleased process group stops the pair without retry. All partial logs and
records remain on disk. The comparator runs only after both probes pass.

Reproduction after obtaining the exact clones and the SIF wrapper:

```bash
python3 run_native_pair.py --task-root "$TASK_ROOT" --stage "$TASK_ROOT/control/c-startup-science-01"
```

Each individual probe can be reproduced in a fresh execution directory with the
exact argv recorded in `execution/state.json`. Do not rerun into an existing
execution directory. This diagnostic does not modify source files or a shared
GPU, and it does not reuse or reset another worktree.

## Other evidence and limits

Agent A's missing-dependency environment independently reports 169 registry
metadata/signature matches, 169 per-loss requirement matches, 40 PyRosetta loss
classifications, and matching Python/NumPy/Torch states, including retained
ImportError cause/message. Its original recovered tests cover cached failures,
concurrent first use, complete symbol publication, and existing monkeypatch
contracts. Those contract checks are complemented here by actual installed
native execution. The native probe does not certify the numerical output of
every registered PyRosetta loss, checkpoint/restart, GPU execution, or MCMC
acceptance. The separate scientific positive CLI control still belongs in the
final integrated-source gate.

## Independent completed W0 audit: c90 versus 9d

**Audit DONE, no scientific/configuration/release discrepancy; no stable
additional performance gain.** The unchanged independent reader (SHA
`eda968172261e7b266278a4616c65d17b8b70582f3e670b41b7249dff45f5a3a`) was reused
with only exact expected HEADs and the native-host artifact layout changed in
memory. `review_reused_w0.py` records those two adaptations and provides a
reproducible command. No new benchmark ran. The raw archive SHA is
`1648312fcb0833d9821a863e9cf131f1813c7ecdbb9bd4e791f5658318818a7d`.

The audit reads all 634 original files: 30 trials, 90 completed trajectories,
270 actual evaluator calls, 30 full trial configurations and 90 full executed
configurations. It checks 436 baseline and 437 candidate source-file hashes
against the exact Git objects. The four measurement modules and full scientific
configuration/budgets match prior CI04. All scientific CSV bytes match for each
repeat/seed. All 20 resident terminal broker ledgers are empty; every observed
process tree is released. Timing reconstructed from raw telemetry and completion
timestamps agrees within 2.05 microseconds.

| Mode | Aggregate effective throughput change | Faster pairs | Disjoint ranges |
| --- | ---: | ---: | --- |
| direct | +4.2869% | 4/5 | No |
| resident_serial | −0.1072% | 3/5 | No |
| resident_adaptive | +0.8966% | 3/5 | No |

The paired median gains are respectively +7.4380%, +0.1394% and +0.9198%; they
are different statistics from total completed trajectories divided by total
observed time. All timing ranges overlap. Candidate resident modes remain about
3.344× and 3.400× slower than candidate direct on the paired median. The original
20% throughput objective is not met by this candidate. The broader Runtime
change is supported by correctness/ownership tests, not a speedup claim.

The 90 accepted CSV rows are initial evaluations. All 180 actual mutation
attempts were rejected, with zero accepted mutation or stage-entry rows; neither
the initial rows nor evaluator throughput demonstrate scientific progress.
W0 contains no structures or full RNG snapshots. Its successful parity does not
replace the native RNG/geometry control above or the accepted-mutation/stage
positive control. Process release is limited to the observer's recorded scope.

## Reanalyse without running any workload

From the committed evidence directory, extract into fresh directories and use
the project environment with OmegaConf for the W0 reader:

```bash
mkdir native-raw w0-raw
tar -xJf native-raw.tar.xz -C native-raw
tar -xJf w0-c90-raw.tar.xz -C w0-raw
python compare_native_startup.py native-raw/execution/baseline.json native-raw/execution/candidate.json --output native-comparison-reproduced.json
python review_reused_w0.py --reader ../review_r1_ci.py --repo "$REPO" --artifacts w0-raw/w0-reuse-c90-vs9d-01 --prior w0-prior-ci04 --out w0-reproduced
```

`$REPO` is a Git clone containing commits 9d62b3e and c90fadfd. The reader rejects
a changed reader SHA, wrong source objects, changed configurations or measurement
modules, failed/incomplete work, mismatched scientific rows, overlapping trial
execution, inconsistent timing and incomplete observed release. The included
CI04 plan is the original prior-plan bytes, used only to compare frozen workload,
measurement hashes and budgets. Original native commands and source identities
are inside the raw archive's `execution/state.json` and native result files.

## Preserved review observations

`review-auxiliary.tar.xz` preserves five full diagnostic/transport records with
`review-auxiliary-manifest.json` giving their individual SHA-256 hashes. Extract
with `mkdir review-auxiliary && tar -xJf review-auxiliary.tar.xz -C review-auxiliary`.
This includes the initial moving-worktree observation and the corrected immutable
Git-object audit context; no unsuccessful measurement or scientific comparison
was silently replaced. The original W0 performance negative is preserved in full.
