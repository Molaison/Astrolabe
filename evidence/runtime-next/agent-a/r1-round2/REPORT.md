# R1 round2: bounded follow-up and stop rationale

**Recommendation: STOP and freeze the validated production HEAD.** The completed
R1 experiment supports a repeatable scoped CPU improvement, while the 20% wall
reduction target remains PARTIAL. The remaining import tree contains one narrow
optional cost, `Bio.SeqIO`, but no newly identified dominant unnecessary
dependency that justifies delaying the final end-to-end gate.

## Actual R1 outcome and remaining phases

The coordinator downloaded complete CI `41764768462` evidence to
`control/cpu-ci-03`. Its immutable candidate is
`71c7592e27d02b2772d984d9b1f5ed9fea47fd6c`, baseline d66. Reading that report and
its traces adds no work to the frozen 30 trials. Every same-mode pair is faster
(5/5 in each mode), ranges are separated, and scientific/release gates pass.
Resident median wall time drops from 5.4685 to 4.7865 s (serial) and 5.4677 to
4.7920 s (adaptive): approximately 12.4%, below 20%.

`r1-ci-decomposition.json` recomputes phases from this run's raw JSONL and records
the source report SHA. Serial pre-broker median is 0.9721 → 0.8205 s; median
trajectory duration is 1.3707 → 1.2067 s. Candidate time before the first stage
remains 1.1322 s per trajectory, while its three-evaluation stage span is
0.0153 s and total measured stages per trial are about 0.0010 s. This is a
startup/control overhead observation, not evidence for altering scientific work.

## Bounded cold-load investigation

An isolated `import-venv` installs the exact `requirements-cpu.txt` versions,
without system packages, Torch, Pandas or RDKit. Its Python is **3.12.14**;
CI uses **3.11**. Local import times therefore locate dependencies and must not
be subtracted from CI wall times or presented as expected benchmark savings.
The parent/shared venv and every production file remain unchanged.

The budget permits at most four cold invocations, each with a 30 s timeout and
**zero scientific evaluations**. Three were used: one diagnostic invocation
failed because resolving the venv executable symlink selected the base Python
without Hydra; the corrected profile and transparent import-call observer both
succeeded. The failed stdout/import log is retained. `budget.json` records actual
return codes and combined observed command wall time; dependency installation is
recorded separately in install logs/report and a package freeze, not counted as a
performance observation.

One real `_load_run_module` profile at integrated source
`c485d6baa15f8a6ae66b5a1bccb55592f1e07745` took 0.5365 s. JAX, Torch and SciPy were
not imported. Large cumulative import entries were Hydra 91 ms, NumPy 72 ms,
FileLock 53 ms and `Bio.SeqIO` 82 ms. Cumulative entries overlap; they are not
additive costs. Hydra/config composition, NumPy sampling/scores and FileLock run
directory ownership are used by W0 and must remain.

## Only plausible remaining import candidate

The observed caller is `_load_run_module -> run.py -> argparser ->
protomer_config_parser -> msa_utils`. `msa_utils.py` imports `Bio.SeqIO` at module
scope; its sole local consumer is `parse_a3m_sequences`. The only other Astrolabe
SeqIO import is already local to `utils.file_parser.parse_fasta`. Transparent
observation confirms this single application startup entry. `Protomer.__init__`
does not parse homology when it is None, as in frozen W0.

A future candidate could move the existing optional import into
`parse_a3m_sequences`, retaining `SeqIO.parse(..., "fasta")`, IDs, insertion
removal, gaps and the exact missing-Biopython RuntimeError. It requires no new
parser, cache, worker pool or lifecycle mechanism. Risks to verify are the error
boundary and first-import Python/NumPy/Torch RNG state, with existing RNA/AF3 and
antibody-framework MSA fixtures. No such code change was made in this round.

The full 82 ms is not avoidable: `Bio.PDB.StructureAlignment` subsequently needs
`Bio.Align`/Seq classes, a subtree representing about 32 ms in this observed
profile. Moving SeqIO would redistribute those shared costs. The potential
exclusive saving is therefore only tens of milliseconds per local cold process,
plausibly a few percentage points end to end, **not a measured CI prediction**
and not evidence that the remaining 20% target can be reached. Expanding into
all structural imports or process reuse would enlarge ownership/scientific
validation scope for an uncertain small gain. That is the reason to stop here.

Raw install/profile/caller/CI-decomposition evidence is preserved losslessly in
`r1-round2-raw.tar.xz`; `manifest.json` lists every file's SHA and the archive SHA. The
two observer scripts are also readable beside this report. No new scientific
workload, mocked speed result, GPU operation or production modification was used.

The initial local diagnostic directory was named `r2-investigation`; original
command records retain that path. The coordinator names this follow-up **R1
round2** to distinguish it from backlog item R2. Renaming the evidence changes
no executed command, measured value or scientific work count.
