# R1: spatial loss imports — candidate awaiting the frozen CI experiment

## Status and hypothesis

Production candidate `605bad0` and CI wiring `9ddd39e` are reviewable commits.
Local behavior/source gates are **DONE**. Performance acceptance is **PARTIAL**
until the unchanged 30-trial comparison completes on one real IPC/PID-capable
runner. There is no GPU claim, and the existing 20% lower-overhead objective is
not certified by an import-only probe.

The completed first CPU CI run (`41760526028`, candidate `6620da6`) showed no
credible speedup from the correctness fixes: candidate direct median 2.8286 s,
resident serial 7.2894 s and adaptive 7.3085 s, with overlapping baseline ranges.
`cpu-ci-01-decomposition.json` derives only from that run's raw trace. Candidate
serial median pre-broker interval is 1.378 s; each trajectory lasts 1.824 s, with
1.715 s before its first recorded stage. Its three-evaluation stage span is
0.0202 s and summed measured CPU stages are 0.00120 s. These disjoint
observations locate cold startup as a candidate, not a proven optimized result.

Caller path is `Astrolabe.run -> _load_run_module -> run.py -> decision_maker ->
losses.base -> losses/__init__.py` discovery. The registry eagerly imports all
loss modules. Spatial dependencies therefore loaded even for W0's sequence-only
loss. Broker/guardian owners, subprocesses, seeds, repeat boundary stop handling,
scientific algorithms and evaluation counts remain unchanged.

## Attempts and preserved negatives

| Evidence | Actual result | Interpretation |
|---|---|---|
| `r1-red.log/xml` | Collection failed because RDKit was missing | Environment dependency failure; **not** an optimization regression |
| `r1-import-red.log/xml` | Original source fails cold import guard | Baseline actually loaded `scipy.spatial` |
| `r1-light-green.log/xml` | Intermediate two-module attempt: 43 pass, cold guard fails | Negative attempt, despite historical filename; no performance keep |
| `r1-residual-import.log` | Actual import stack identifies `design_screening.py`'s indented optional import | Third entry point required; not guessed from import duration |
| `r1-w0-before/`, `r1-w0-after/` | Both real direct 3×3 W0 runs succeed; spatial still loaded | Two extra correctness/import diagnostics, **18 evaluations total**, no timings/throughput, not CI trials or warmup |
| `r1-w0-diagnostic-comparison.json` | All scientific fields exactly equal; 141 available loss metadata entries equal | Intermediate candidate only; limited dependency environment |
| `r1-geometry-missing-torch.log/xml` | Independent geometry environment still needed Torch via `utils.chemicals` | Dependency failure preserved; no fake module substituted |

The two-module intermediate patch was never committed or published. The accepted
experiment candidate additionally moves `design_screening`'s import into its
existing `_require_ckdtree` helper. That helper returns the same SciPy class and
retains its explicit ImportError message when spatial support is unavailable.
All three mathematical call sites use that returned class.

## Correctness and dependency scope

`interaction` and `repeat_symmetry` now import SciPy locally at their actual
geometry functions. `interaction` originally did not postpone annotations: only
the private Delaunay return annotation is quoted and its type import is guarded
by `TYPE_CHECKING`; public annotations are unchanged. Delaunay and QhullError are
both locally available in the original try/fallback scope. The existing Qhull
test injects its first-call error at `scipy.spatial.Delaunay` and still requires
the same fallback options and occupancy result.

The independent geometry environment is
`/workspace/scratch/1a0b1639644d/cloud-20261104/geometry-venv`, Python 3.12.14.
RDKit 2025.9.6 and Pandas 2.2.3 come from PyPI; CPU Torch 2.7.1+cpu comes from
`https://download.pytorch.org/whl/cpu`. Exact install URLs/hashes and dependency
versions are recorded in `geometry-install-report.json`,
`geometry-torch-install-report.json`, install logs and
`geometry-pip-freeze-final.txt`. The shared venv and frozen W0 dependencies were
not changed. The pinned RDKit release provides Linux Python 3.11/3.12 wheels:
<https://pypi.org/project/rdkit/2025.9.6/>.

The unchanged original structure suites plus new real PDB/SciPy clash, hotspot
and B-factor controls pass: `r1-geometry-green` **67 pass / 0 skip**. With original
production source in `agent-a-r1-before`, the same controls pass and the cold
guard fails as expected: `r1-geometry-baseline` **66 pass / 1 failure**. The
baseline retains its original Qhull injection test. These tests do not certify
PyRosetta execution or pretrained model inference.

The exact seven-suite command, run separately in candidate and baseline
checkouts with the same geometry interpreter, was:

```sh
/workspace/scratch/1a0b1639644d/cloud-20261104/geometry-venv/bin/python -m pytest -q \
  tests/test_loss_import_lifecycle.py tests/test_design_screening_geometry.py \
  tests/test_interaction_ipae.py tests/test_repeat_symmetry.py \
  tests/test_ligand_reference_pose_distance.py tests/test_sequence_losses.py \
  tests/test_loss_registry_chain_pair_losses.py --junitxml=OUTPUT.xml
```

The attached JUnit establishes the scope precisely:

| Suite | Cases |
|---|---:|
| `test_loss_import_lifecycle.py` | 1 |
| `test_design_screening_geometry.py` | 4 |
| `test_interaction_ipae.py` | 3 |
| `test_repeat_symmetry.py` | 15 |
| `test_ligand_reference_pose_distance.py` | 16 |
| `test_sequence_losses.py` | 25 |
| `test_loss_registry_chain_pair_losses.py` | 3 |

Root independently ran the first five suites and reported **39 passed** in
`evidence/runtime-next/review/spatial-root.log`. Thus **67 = 39 + 25 + 3**;
these are separately executed gates, not 106 distinct passing tests or one
combined run. `r1-test-scope.json` retains the per-suite counts parsed from A's
JUnit. The baseline checkout contains original production source and the two
new guard/control files; its original Qhull test uses its original binding.

`r1-registry-before.json` and `r1-registry-after.json` prove all **169** available
loss names and metadata are equal in the full geometry environment. The candidate
cold runner no longer loads `scipy.spatial`; first lazy use preserves Python,
NumPy and installed Torch CPU RNG state. No global registry loader was replaced.
`check_r1_source.py` verifies numerical bodies/decorators are AST-identical after
normalizing only allowed import/type/helper wiring changes. Dependency failure
behavior has a separate controlled missing-import test.

## Measurement gate

`9ddd39e` adds the cold import/RNG guard to the ordinary W0 regression gate and
adds real geometry suites to the separate CPU Torch job. Only that separate job
installs `requirements-geometry.txt`; W0's job configuration and dependency file
remain byte-for-byte unchanged. The local harness gate is 26 pass. Measurement
functions, source baseline d66, the frozen protocol hash, 5 groups × 2 revisions
× 3 modes, 9 evaluations per trial, 900 s comparison budget and 260 MB output
budget remain unchanged. Same-mode paired differences and ranges determine
whether to keep/revert. The candidate must also be compared with the best direct
mode and must not count failed/incomplete work as speed.

At evidence packaging, root reports publishing candidate `71c7592` and starting
CI run `41764768462`. No result from that in-progress run is claimed here.
