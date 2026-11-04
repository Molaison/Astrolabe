# B independent interrogate: R1 lazy spatial imports

**Verdict: PASS for the reviewed correctness and import boundary. Performance acceptance remains outside this review.** No reproducible production defect was found. Reviewer B did not author or modify the three production modules.

## Exact scope

| Item | Exact value |
|---|---|
| Candidate | `605bad02f5c71083a95142f03bebc0db24a27d84` |
| Baseline parent | `cf62f856cf4dfde861a7b0d2589aa843ea53f79e` |
| Independent candidate checkout | `cloud-20261104/review-b-a-r1` |
| Independent baseline checkout | `cloud-20261104/review-b-a-r1-before` |
| Environment | `cloud-20261104/geometry-venv/bin/python`; Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, RDKit 2025.09.6, Torch 2.7.1+cpu |
| Production owners | `astrolabe/losses/{interaction,repeat_symmetry,design_screening}.py` |

The candidate tree stayed clean. The baseline production tree stayed unchanged; only the two new test fixtures (`test_loss_import_lifecycle.py`, `test_design_screening_geometry.py`) were copied from the candidate. Its original ligand/Qhull test was retained, so it patches the original dependency location. The three production modules at this parent are also byte-identical to the frozen `d66ee714f2e755948534f95d9d125287ce8dbd91` baseline (`git diff --quiet` exits 0). Tests ran in the existing isolated geometry environment; the frozen W0 environment was not changed.

## Reproduced evidence

| Gate | Baseline | Candidate | Evidence |
|---|---|---|---|
| Seven geometry/import/registry suites | 66 passed; only expected cold import guard failed | **67 passed, 0 skip** | `review-a-r1-{before,after}-tests.log`, corresponding XML and command JSON |
| Independent full-runner cold import guard | Fails, exit 1: `scipy.spatial` already loaded | Passes, exit 0; loads spatial only at first real geometry call | `review_a_r1_probe.py`, `review-a-r1-{before,after}-probe.json` and logs |
| Complete installed loss registry | 169 entries | Same 169 metadata, defining modules, and public signatures | Raw probe JSON; `review-a-r1-comparison.json` |
| Independent real SciPy/Bio.PDB numerical cases | 64 values and 13 expected ValueErrors | **All 77 records exactly equal** | Raw probe JSON, including float hexadecimal values, ndarray dtype/shape/values, and error type/message |
| Global Python, NumPy, Torch CPU RNG | Same seeded states | Unchanged at first lazy SciPy call and after every numerical/error case | RNG state hashes in raw probe JSON |
| Missing spatial dependency, three design-screening callers | Explicit ImportError before reading the sentinel file | Same ImportError type/text; explicit dependency `__cause__` added | `review-a-r1-{before,after}-missing.json` and command JSON |
| Whole-module source equivalence | Original AST | Identical after reversing only individually asserted import/helper wiring changes | `review_a_r1_source.py`, `review-a-r1-source-equivalence.json` |

The independent numerical probe uses its own coordinate/PDB fixture, not the author's test fixture. It covers CA strict-boundary versus sidechain inclusive-boundary contact behavior; same, absent and cross-chain selections; malformed/masked/nonfinite inputs; heavy/CA repeat contacts and minimum distance, empty inputs and invalid atom modes; clash counts, hotspot residues and B-factor means; interface residue selection; real Delaunay tetrahedra and irregular points; duplicate, nonfinite, radius-limit and coplanar cases. Actual SciPy and Bio.PDB execute throughout. Synthetic coordinates are a correctness control, not a pretrained inference workload.

Registry digest (canonical sorted JSON): `986ac4188a1fc650f02ab7b361defd2744c0536090f0ec06bf03106f7954ffc8`. Numerical digest: `00d827560b4fe35f043ec21089eaefb4b76403344b54b285fd26c21ab51c6b24`.

## Owner and semantic review

The registered function signatures and all complete-module mathematical/decorator/default ASTs are preserved. The source guard reverses a fixed list of nine local imports, the exact `TYPE_CHECKING`/quoted private Delaunay annotation, the exact checked `_require_ckdtree` implementation, and three local bindings to that helper. It does not generically strip arbitrary function bodies or all exception handling.

The private `_alpha_shape_kept_simplex_indices` caller at `interaction.py` uses its returned triangulation directly. Repository search found no runtime `get_type_hints` consumer of this private helper. The type-only import plus quoted annotation avoids eager SciPy loading; this review does not establish a new private annotation-introspection API. The registered/public signatures are unchanged.

For installed dependencies, registry equality is verified for all 169 entries. Under the explicit **unavailable SciPy** fault, the baseline has 117 entries and the candidate 169: deferred dependency failure lets loss discovery finish. This changes the visible registry and failure timing; it must not be described as registry equality in every missing-dependency environment. Previously unavailable geometry does **not** become executable without SciPy. The three verified design-screening entry points, `_calculate_bindcraft_clash_score`, `_bindcraft_hotspot_residues`, and `halludesign_key_res_plddt`, still raise the same explicit ImportError before structure I/O. The additional exception cause identifies the failing dependency. The raw before/after exception types, messages and cause types are preserved. No regression of a previously executable path was reproduced in the installed environment.

## Preserved negative evidence and limits

The expected baseline cold-import failures are saved. The first independent probe attempt also exposed four **review-harness invocation errors**: three hotspot calls and one interface helper call used positional arguments for keyword-only parameters. Those original records and logs remain as `*-probe-attempt01.*`. They are not production findings and are excluded from the accepted numerical comparison. The corrected probe uses keyword arguments and asserts the exact set and types of its 13 expected numerical errors, preventing an unexpected TypeError from being mistaken for successful equivalence.

All raw `.log` and `.xml` files are stored without byte changes in `review-a-r1-raw-logs.tar.xz`, with entry sizes and SHA-256 in `review-a-r1-raw-manifest.json`. This preserves original pytest failure whitespace. Extract beside this report to restore the paths in the evidence table: `tar -xJf review-a-r1-raw-logs.tar.xz`.

This is not a throughput benchmark, GPU test, pretrained-backend certification, or PyRosetta execution claim. Test duration is only test-run metadata. The main agent's frozen W0 comparison decides performance keep/revert; no import microbenchmark is substituted for it.

## Reproduction

Run from the B worktree, replacing the checkout paths only if needed. Full pytest argv and working directory are retained in the `*-test-command.json` records.

```bash
(
  cd ../review-b-a-r1
  ../geometry-venv/bin/python -m pytest -q \
    tests/test_loss_import_lifecycle.py \
    tests/test_design_screening_geometry.py \
    tests/test_interaction_ipae.py \
    tests/test_repeat_symmetry.py \
    tests/test_ligand_reference_pose_distance.py \
    tests/test_sequence_losses.py \
    tests/test_loss_registry_chain_pair_losses.py
)
```

The subshell keeps pytest in the candidate checkout so imports resolve to the reviewed code. The independent probes also make repository selection explicit:

```bash
../geometry-venv/bin/python evidence/runtime-next/agent-b/review_a_r1_probe.py \
  --repo ../review-b-a-r1 --out /tmp/r1-after.json --require-lazy
../geometry-venv/bin/python evidence/runtime-next/agent-b/review_a_r1_probe.py \
  --repo ../review-b-a-r1-before --out /tmp/r1-before.json --require-lazy
# The baseline command intentionally exits 1 after saving its complete numerical result.
../geometry-venv/bin/python evidence/runtime-next/agent-b/review_a_r1_probe.py \
  --repo ../review-b-a-r1 --out /tmp/r1-missing.json --missing-spatial
../venv/bin/python evidence/runtime-next/agent-b/review_a_r1_source.py \
  --repo . --out /tmp/r1-source.json
```
