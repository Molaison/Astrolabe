# Agent A independent final-source and allocation audit

**DONE:** published source identity, equivalence to tested production, frozen measurement integrity, and terminal accounting for the sole GPU request. **BLOCKED:** actual pretrained GPU execution; no allocation or inference occurred.

## Source proof

Published GitHub commit `1bdaf3a53227c9197b930dc00a01b752acd90067` and local `5db26b44cdf2d229ca35558f611b8b645ead881b` share tree `6f763bfb04778ef4f32386ddb8aeb38ce4240bf1`. An independent GitHub GET, local Git object comparison, and before/after clean working-tree/index checks establish that identity.

All 948 production paths in the recorded scope match tested `f9ce87b9f5cf37710f0a52ebe503595e5d7fdd7f`: 917 blobs also match current filesystem SHA256, and 31 Gitlink commit references match. This includes all 275 `astrolabe/**` paths. Gitlink equality does not claim that submodules were materialized. The four frozen measurement modules are byte-identical to d66, f9 and final HEAD. Three complete scoped diff files are empty.

The full repository differs from f9 in validation/launcher/tests/docs/evidence, as explicitly inventoried. `source-audit.json` and `frozen-measurement-manifest.json` provide concise proofs; the complete manifest, GitHub response, script, diff files and original preallocation scheduler captures are in the unchanged `preallocation.tar.xz` archive.

## Sole GPU request

Root submitted exactly one new request, `33334`, with name `astrolabe-gpu-20261104-10cf2b34c104`, under UID7054 / xulab / boron / normal. It queued at 05:51:06 UTC and terminated at 05:52:06 UTC on 2026-11-04 after `srun` reported the requested nodes busy.

Independent exact-job `sacct`, `scontrol` and `squeue` reads confirm `CANCELLED by 7054`, no actual start, ElapsedRaw0, AllocCPUS0, empty AllocTRES, no allocated node and no active queue row. The launcher directory lacks `allocation.json`; the original stderr and submission JSON were independently read and byte-hash verified. **Actual allocated GPUs: 0. Actual allocation GPU-hours: 0.** The 0.25 GPU-hour request ceiling is not usage, and the approximately 60 s waiting interval is not GPU occupation.

The actual GPU workload remained BLOCKED. No performance ratio, throughput improvement, scientific GPU result or resource release from a running GPU process is claimed. No retry, manual cancellation, shared GPU query, attach to another job or resource-policy change was performed by Agent A. The terminal request needed no further cancellation.

`allocation-proof.md` and `accounting-proof.json` preserve the precise evidence and limits, including the future scontrol StartTime that must not enter elapsed-time arithmetic. Current node headroom is only an arithmetic observation; the record's Priority reason and per-GPU memory fields are retained without inventing a unique scheduler cause.

## Lossless raw evidence

| Archive | Files | Bytes | SHA256 |
|---|---:|---:|---|
| `preallocation.tar.xz` | 12 | 80628 | `cdf243d4634ec6456fa8241578d90591add2950ec68c31ad788a5b444f09a77b` |
| `postallocation.tar.xz` | 13 | 9664 | `f855e9c6a53f272237a24429d89e33409c04e4d7b290e4f84317833f32da74cb` |

Each archive has a separate per-file SHA256/size manifest. Every postallocation archive member was re-read and rehashed after compression. The negative first report-parser run (slash-bearing scontrol keys) is retained; only the evidence parser changed, with no source or scheduler mutation. B separately owns the complete GPU preparation/raw directory archive; it is not duplicated here.

Extract each archive into a fresh directory to inspect original reports and scripts. The postallocation audit can be rerun as `python job-33334/audit_accounting.py`; it only reads captured records and writes its local proof JSON. Source audit reproduction and scope are documented inside `final-head-a/README.md` in the preallocation archive. No additional scientific evaluation or GPU execution is required for either audit.

This commit is evidence only, based directly on immutable source `1bdaf3a`. It changes no production, measurement, launcher, validation or test source file.
