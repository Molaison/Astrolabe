# Root f102 independent verification

The root separately reproduced the admitted waiting leak and GPU checker
identity/release defects, and independently reran the native first-use comparator
against complete unmodified C raw evidence (28 checks PASS). The new W0 audit
uses direct CSV equality, actual per-trial JSONL phase counts and process release
reconciliation, plus fresh timing aggregation: 30 valid trials, 90 trajectories,
270 actual cpu_execute events, 90 accepted initialization rows and 180 rejected
mutation attempts. All 15 performance pairs improve; ranges separate.

The launcher UUID/ordinal configuration-path probe and A's original metadata
probe were independently rerun before and after the fixes. Metadata control
passed 2/7 before and 7/7 after, with no subprocess/GPU invocation. These pure
controls do not claim an actual wrong-device run or an allocation certificate.

A combined local regression invocation in the intentionally minimal CPU venv
produced 90 passes and eight missing-RDKit failures. The unchanged suites then
passed **98/98 in the existing geometry environment**, zero skips. The mandatory
lazy-import suite was assigned to the existing Torch/geometry CI job; no frozen
CPU benchmark dependency or scientific workload changed. The negative JUnit is
retained. An earlier incorrect test filename caused collection exit4 before any
test; this was corrected to the observed test_runtime_validation_harness.py.

The resource snapshot is read-only scheduler data at 05:15:33 UTC; it is not a
Slurm allocation or GPU proof. Actual device/time/release must come from the
separate bounded job. No existing user job was touched.
