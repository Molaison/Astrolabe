# Agent B: final-source CPU readiness and blocked GPU admission

**DONE:** immutable source/CPU/preparation evidence and independent reread. **BLOCKED_ADMISSION:** the single GPU request never obtained resources. **GPU NOT_RUN:** no pretrained GPU correctness, residency, reuse or throughput certificate. Actual allocated GPU-hours are zero, established independently by Agent A and reread by B.

The human report is [runtime_gpu_mutator_certification.md](../../../../docs/runtime_gpu_mutator_certification.md). The current machine-readable narrow scope is [runner-certification-supplement.json](../../gpu-control-01/runner-certification-supplement.json). Registry rows remain 28 BLOCKED / 6 PARTIAL.

## Preserved artifacts

| Artifact | Contents |
|---|---|
| `gpu-admission-blocked-33334.tar.xz` | All original `gpu-control-01/`, `gpu-allocation-01/`, and `control/gpu-execution-root-01/` files, including partial/failed logs and warnings. |
| `archive-manifest.json` | 27 original files / 977,895 bytes / 40 file-directory entries, with SHA-256, sizes and original metadata. The archive also contains this manifest. |
| `archive-verification.json` | Complete bounded transport, final archive digest and every original-file digest verified. No execution/trial/inside-allocation record exists. |
| `source-handoff-verification.json` | Exact published source HEAD/tree and agreement with the independently reviewed harness/launcher/protocol/native source. |
| `native-freeze-review.json` | Actual CPU constructor/input manifest and all 302 source digests; the subsequent tensor reread below closes this earlier review's pending parsed-input transfer. |
| `native-launch-review.json` | Actual CPU-only preparation, identical environment and precise HOME limitation. |
| `cpu-and-admission-review.json` | CPU reload of the original complete input and independent parse of A's exact-job accounting captures. No model construction or native/GPU command. |
| `independent-review-artifacts.tar.xz` | Exact B transport records, all source digests, prepared commands and `audit_cpu_and_admission.py`; these are review artifacts, not runtime changes. |
| `independent-review-manifest.json` | Every sealed review artifact's size and SHA-256. |
| `files.sha256.json` | Digests of the standalone files in this directory, excluding that manifest itself. |

Raw archive SHA-256: `5588acc27dceb7595fe7aff8be79385ec9186913aa7a486fc54e8ea0e750b8e9` (126,328 bytes).
Review archive SHA-256: `34c151b454db0a882ddf7bfb4c145281a50b1bfadd7616889daaa27ff4e5284d` (184,192 bytes).

The original launcher status `PARTIAL_ACCOUNTING_OR_BUDGET` is preserved. [Agent A's terminal proof](../agent-a/accounting-proof.json), backed by [complete raw accounting captures](../agent-a/postallocation.tar.xz), independently establishes CANCELLED/no start/no resources/zero allocation. B did not query or modify Slurm. `Requested nodes are busy` is an admission outcome, not proof that all physical devices were occupied.

## CPU-only reproduction

Use a fresh review directory and a separate checkout of the frozen source at `1bdaf3a53227c9197b930dc00a01b752acd90067`. Extract the original archive under `review/raw`, and extract only `audit_cpu_and_admission.py` from the review archive into `review/`. Extract Agent A's `postallocation.tar.xz` under `review/accounting/`; it yields `job-33334/`.

Then run with a CPU Python environment containing Torch 2.7.1 and the repository's codec dependencies:

```text
python review/audit_cpu_and_admission.py review /absolute/path/to/frozen-source review/accounting/job-33334
```

The script reads original bytes, validates source/input identities, reloads the non-pickle tensor codec, checks all 354 positions/masks/native sequence, and parses the recorded accounting responses. It performs no model construction, forward/sample, container, Slurm or GPU operation. Its environment explicitly requires a CPU-only Torch build. Do not replace the frozen source with a later evidence-only HEAD and call that a native rerun.

There is no GPU `execution.json`, trial directory or comparison to replay. Do not create a synthetic comparison in their place. Actual GPU execution and repeated effective-throughput improvement remain blocked.
