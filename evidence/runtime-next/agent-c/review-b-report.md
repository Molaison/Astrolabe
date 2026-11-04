# Independent interrogate: ESMFold output boundary

**Reviewer:** Agent C, independent of author B. **Exact commit:** `a55c28553917742ed451aa016ce7dc62cdfea95c`. **Verdict:** PASS for the inspected CPU/native-output and handler boundary contracts; no reproduced new production defect in this scope. GPU/pretrained correctness, actual D2H traffic and effective workflow throughput remain **BLOCKED/unverified**. Independent CPU output-only timings are negative and retained.

## Isolation and scope

The reviewed commit was checked out in a fresh detached worktree `review-c-b-a55c285`. It stayed clean throughout review. B's concurrent certification-matrix work was not included. C did not modify production code or the authored tests. The inspected change comprises the new `esmfold_output.py` boundary, its code-path inclusion in resident identity, and the daemon's explicit device/host ordering.

Actual torch CPU tensors and the exact source-extracted native atom mapper/Protein/PDB formatter were executed. Model outputs were synthetic. The handler contract executes the production handler AST with a recording lease and real tree codec. None of these are a real model-weight load, inference, broker/UNIX transport, CUDA lease, GPU numerical test or transfer-bandwidth measurement.

## Independent checks

| Check | Observed result |
|---|---|
| Author's final focused tests | **63 passed, 1 deselected in 2.35 s**; excluded test requires the locally unsupported process observation path |
| Same actual-handler gate against old d66 daemon source | Expected failure: `unused device output retained past lease` |
| Same handler gate against a55c285 | Pass |
| C's independently constructed CPU differential | **36/36 cases pass:** float16/float32/float64; batch 1/3; residues 2/7/31; padded/non-padded; noncontiguous positions and PAE |
| Scientific comparisons in those cases | Exact PDB bytes; score tensor dtype and bits; untouched input tensors; identical Python/NumPy/torch RNG state |
| C's additional compatibility negative controls | Changed NumPy binding, residue table, formatter dependency and custom wrapper all refuse split; restored source accepts |
| Final source/fixture hashes | Committed fixture integrity and actual source gate pass |

The independent differential is `probe_esmfold_differential.py`; its raw 36-case hashes and results are in `review-b-independent-differential.json`. It does not import or reuse the author's test-output generator.

## Code and scientific dependency assessment

The optimized path is enabled only after checking the actual bound wrapper and native call graph by normalized source hashes, plus relevant atom/residue constants and library bindings. It keeps native atom14→37 mapping on the original tensors inside device admission, materializes the required field union using the existing dtype-preserving `cpu_tree`, captures RNG, and drops unused output references before lease exit. Host formatting calls the same native Protein/to_pdb routines on CPU arrays. Shared score/PDB fields are materialized once.

Unknown/modified/decorated helpers use the original composite native path. That preserves custom formatter behavior and its RNG consumption before capture; the author's real-tensor handler test exercises a custom formatter that consumes Python/torch RNG. Missing chain data preserves the original native error path. The host finalizer rejects a non-CPU tensor and its valid path does not issue transfer calls. The new helper's source is included in resident worker identity, so changing it invalidates the recorded implementation identity.

This is a narrow compatibility argument for the pinned helpers and tested outputs, not complete certification of all ESMFold distributions, arbitrary model outputs or GPU precision modes. Runtime monkey-patching is explicitly unsupported; helpers are resolved once for an immutable worker session.

## Independent performance negative result

C reran the unchanged benchmark harness on the detached commit: 7 alternating trials per implementation/shape, one torch CPU thread, zero errors, exact PDB/score/RNG controls. Requested host-materialization counts reproduce 21 → 9 calls. The counted bytes are tensor payload sizes requested by `.cpu()` / `.to('cpu')`; inputs already reside on CPU, so these are **not actual GPU transfer bytes**.

| Synthetic batch × residues | Requested host bytes, baseline → candidate | Median output-only CPU ms, baseline → candidate | CPU time change |
|---|---:|---:|---:|
| 1 × 24 | 635,728 → 20,648 | 1.1807 → 1.2907 | +9.31% |
| 2 × 128 | 34,256,928 → 326,672 | 10.1178 → 10.7941 | +6.68% |

Both independent CPU medians are slower. This repeats the direction of B's original +1–3% negative result, with a larger observed slowdown in this run; no confidence interval or universal slowdown magnitude is asserted. Keeping the change can be justified by the validated ownership boundary and deterministic reduction in requested host payload, **not** by a demonstrated CPU/GPU throughput win. Actual inference, startup, broker scheduling and end-to-end effective throughput are outside this output-only sample.

Raw interleaved trials, byte events and science checks are in `review-b-output-comparison.json`; original B measurements and negative results remain in its committed evidence. No negative data was overwritten.

## Findings and discarded claims

No new production defect was established by the final-commit review. One source-gate failure was observed in B's work-in-progress while fixture whitespace/source-hash normalization was being corrected; the gate conservatively returned None. That observation was sent to B, the author retained its negative evidence, and the exact final commit passes. It is not reported as a defect in a55c285.

Claims rejected as unsupported: “reduced requested host payload proves GPU speedup,” “CPU code-block timing proves CUDA utilization,” or “native synthetic output parity certifies pretrained backend inference.” Actual GPU, weights and full process/broker certification remain separately recorded gaps.

## Reproduction and artifacts

From the detached reviewed checkout:

```sh
../venv/bin/python -m pytest -q tests/test_esmfold_output_boundary.py tests/test_tree_codec_torch.py tests/test_evaluator_rng_lifecycle.py -k 'not worker_preserves_constructor_dtype_cudnn_and_native_precision_side_effect'
../venv/bin/python evidence/runtime-next/agent-b/esmfold_worker_negative_control.py --baseline
# Expected exit 1.
../venv/bin/python evidence/runtime-next/agent-b/esmfold_worker_negative_control.py
# Expected exit 0.
../venv/bin/python ../agent-c/evidence/runtime-next/agent-c/probe_esmfold_differential.py --repo . --out /tmp/c-esmfold-differential.json
../venv/bin/python evidence/runtime-next/agent-b/esmfold_output_benchmark.py --mode compare --output /tmp/c-esmfold-output-comparison.json
```

Saved C artifacts: `review-b-a55c285-tests.log`, `review-b-worker-{baseline,candidate}.json` and stderr logs, `review-b-independent-differential.json`, `review-b-output-comparison.json`, and the independent differential script. Review involved actual separate agents; it does not claim different model families.
