# ESMFold1 audited output boundary: implementation and CPU evidence

## Outcome and scope

**DONE:** a backend-specific audited split moves native NumPy PDB formatting out of the device lease, materializes required tensors only once, and drops unused device output references before releasing the lease. Unknown native helpers retain the original composite execution. Source compatibility includes the new helper in the worker identity.

**PARTIAL:** actual torch 2.7.1+cpu tensors, exact source-extracted native OpenFold atom mapping/Protein/PDB functions, score arrays, RNG and the production handler's boundary contract pass. These fixtures contain synthetic model outputs, not pretrained inference. The handler contract executes the production AST with a recording lease and real tree codec; it is not a real broker/socket E2E run.

**BLOCKED:** pretrained ESMFold1 weights, a configured exact model installation, an approved GPU allocation, CUDA execution, actual D2H bytes/bandwidth, GPU numerical certification, and effective workflow throughput. No GPU speedup is asserted.

## Architecture and behavior

The existing daemon remains the sole model owner. `resolve_pdb_boundary(model)` checks the actual bound `output_to_pdb` delegate, the native helper, mapper, gather, Protein type, formatter and its helper dependencies by exact normalized source hashes. Atom/residue naming constants and relevant library bindings are checked as well. Decorated/modified/unknown implementations fail closed to the original native composite call. Assets and imported code remain frozen for a worker session; hot patching is unsupported.

The device phase performs the original native atom14→37 operation on the original tensors, materializes the union of PDB and selected score fields through the existing `cpu_tree`, captures RNG, releases unused output references, and updates lifecycle. The host phase converts only CPU tensors to NumPy and invokes the same native Protein/to_pdb functions. It does not choose a new residue order, chain identifier, mask, rounding rule, scoring tensor, precision, sample count or scientific seed. Required shared fields such as pLDDT and chain_index are materialized once and reused. The existing tree codec and request protocol are unchanged.

Pinned source references are ESM2 `43c67d952da30d2471f609848931d20ec8c98532` and OpenFold `8396098268a5675b0ff776ec274f15d380a4ccf5`. Exact source file/excerpt hashes live in `tests/fixtures/esmfold_output_native.json`; full MIT/Apache license notices accompany the native test excerpts. This is version-gated helper compatibility, not model/version certification.

The pinned formatter unconditionally uses `chain_index.astype`. Missing chain_index therefore already raises AttributeError. Differential tests preserve that error; the worker retains the original native path for this malformed output instead of inventing chain information.

## Measured evidence and negative result

The harness was frozen before implementation. Both baseline and comparison JSON record the same harness SHA. It compares two shape-compatible synthetic output workloads, with 7 alternating trials per implementation, one torch CPU thread, exact PDB hashes and zero errors. The old native helper materializes all output fields, including pair representations and logits, then the daemon materializes scores again. The new boundary materializes only the needed union.

**Requested host-materialization bytes are counted through actual torch Tensor `.to('cpu')` / `.cpu()` calls. Inputs already reside on CPU, so this is a count of requested payload sizes, not a measurement of actual GPU transfer traffic.** CPU timings below are output-only measurements, excluding model inference, startup and broker scheduling.

| Synthetic batch × residues | Requested host bytes | Reduction | Median CPU total, baseline → candidate | CPU time inside the proposed device block, baseline → candidate |
|---|---:|---:|---:|---:|
| 1 × 24 | 635,728 → 20,648 | 96.752% | 0.9202 → 0.9458 ms (+2.78%) | 0.9183 → 0.0801 ms |
| 2 × 128 | 34,256,928 → 326,672 | 99.046% | 8.2038 → 8.3219 ms (+1.44%) | 8.2022 → 0.2045 ms |

The table above records comparison 01 (preserved as `esmfold-output-comparison-01.json`). The CPU total median regressed slightly in both cases in that run. This negative result is preserved; there is no CPU throughput win. The shorter proposed device block is an instrumented CPU code-boundary observation, not measured GPU admission or GPU utilization. The change is retained for the verified boundary contract and large deterministic materialization reduction. Real GPU effective throughput remains unmeasured and is a separate acceptance gate.

The native differential suite checks exact PDB bytes and thus atom/chain order and formatting; float32/float64 score bits; single/multiple batches; multichain, padding and noncontiguous coordinates; original inputs; and Python/NumPy/torch RNG. It rejects changed mapper/formatter/constants, confirms the host finalizer makes no transfer call, and rejects a non-CPU tensor. The actual handler test confirms atom mapping and RNG restore/capture remain inside the recorded lease while native formatting runs outside; a custom formatter that consumes RNG remains inside the fallback lease and produces the original result/state.

Raw test results:

- `esmfold-output-tests.txt`: 23 passed.
- `esmfold-output-regressions.txt`: **63 passed, 1 deselected**; the process worker test is explicitly excluded because local process/socket validation is not reliable here.
- `esmfold-worker-negative.json`: the same real-tensor handler gate against baseline d66 fails with exit 1 (`unused device output retained past lease`).
- `esmfold-worker-control.json`: working implementation passes with exit 0.
- `esmfold-output-run-manifest.json`: base HEAD, exact changed source hashes, raw log hashes and commands bind the pre-commit working-tree run to its implementation.

## Reproduction

From the repository with CPU torch and test dependencies installed:

```bash
python -m pytest -q tests/test_esmfold_output_boundary.py tests/test_tree_codec_torch.py tests/test_evaluator_rng_lifecycle.py -k 'not worker_preserves_constructor_dtype_cudnn_and_native_precision_side_effect'
python evidence/runtime-next/agent-b/esmfold_output_benchmark.py --mode compare --output /tmp/esmfold-output-comparison.json
python evidence/runtime-next/agent-b/esmfold_worker_negative_control.py --baseline
# expected exit 1 against d66
python evidence/runtime-next/agent-b/esmfold_worker_negative_control.py
# expected exit 0
```

The real process resident test is `tests/test_esmfold1_resident.py` (fake model/weights, CPU process contract). It should run on an eligible CI host; success still would not certify pretrained GPU execution.

## Hillclimb decision

One implemented output-boundary hypothesis was evaluated. Keep: required-host-payload reduction and host-formatting boundary, with all correctness gates green. Negative retained: CPU total time +1–3% in this sample; no claimed CPU or GPU speedup. An entire `output_to_pdb` call moved outside the lease is not an acceptable alternative, because its native atom14→37 operation still executes device tensor work. Further kernel/inference or protocol changes are outside this narrow fix and require real configured workload evidence.

## Final source-gate rerun

Vendor source line-end whitespace is normalized identically in the runtime gate and fixture; full unnormalized source-file SHA256 values remain in the fixture manifest. During fixture cleanup, an AST-only hash omitted the formatter's final inline comment and the guard correctly rejected the helper. The 13-failure run is retained in `esmfold-source-gate-negative.txt`; no failing implementation was committed. After correcting the hash against the actual inspected source, the final focused run passed **63 tests, 1 deliberately deselected**.

The final same-harness rerun is `esmfold-output-comparison.json`; its source hashes are in `esmfold-output-run-manifest.json`. The earlier negative CPU timing run and its source manifest are retained separately.

- Final 1 × 24: CPU total 1.1661 → 0.9728 ms (-16.58%); requested host bytes 635,728 → 20,648; output/RNG/input gates pass.
- Final 2 × 128: CPU total 9.3309 → 9.1273 ms (-2.18%); requested host bytes 34,256,928 → 326,672; output/RNG/input gates pass.

Small CPU timings vary across samples and establish no repeatable total-time win. The retained byte reduction is deterministic for these output shapes.
