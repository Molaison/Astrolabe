# Agent A: independent accounting closure for job 33334

**DONE: terminal state and zero allocation independently established. BLOCKED: actual GPU backend execution.** The only submitted request, `33334`, never obtained an allocation. No retry, manual cancellation, GPU query or scientific execution was performed by this review.

| Check | Independent raw evidence |
|---|---|
| Identity | Job `33334`, name `astrolabe-gpu-20261104-10cf2b34c104`, user `zpzeng`, UID `7054` |
| Requested scheduling scope | account `xulab`, partition `boron`, QOS `normal`; 1 g4090 / 4 CPU / 32 GiB / 900 s |
| Submit / terminal time | 2026-11-04 05:51:06 UTC / 05:52:06 UTC |
| Terminal state | `sacct`: `CANCELLED by 7054`; `scontrol`: `CANCELLED` |
| Actual start / elapsed | `sacct Start=None`, `ElapsedRaw=0` |
| Actual resources | `AllocCPUS=0`, empty `AllocTRES`, `NodeList=None assigned`; `scontrol` also reports empty NodeList and null AllocTRES |
| Active queue | Exact own-job query returned no rows |
| Client outcome | exit 1; queued, then `Unable to allocate resources: Requested nodes are busy` |
| Client wall | 60.10791897773743 s in the preserved launcher `submission.json` |
| Actual GPU allocation / GPU-hours | **0 GPUs / 0 GPU-hours** |
| Actual GPU inference / end-to-end validation | **0 trials; BLOCKED** |

The 0.25 GPU-hour figure is the requested upper bound, not measured usage. The 60 s submit-to-terminal interval is queue waiting time, not GPU occupation. There is no actual start from which to calculate `End − Start`; `scontrol` retains a future timestamp (2026-11-05 02:44:35), which is explicitly excluded from elapsed-time arithmetic. The accounting exit code `0:0` is retained as raw data and is not interpreted as successful workload execution: the state is CANCELLED, the `srun` client exited 1, and no allocation occurred.

The launcher directory has no `allocation.json` or `inside-error.json`; its stdout is empty. Complete original `submission.json` (1347 bytes), stderr (117 bytes) and stdout (0 bytes) were independently read and SHA256-verified. Launcher status remains its recorded `PARTIAL_ACCOUNTING_OR_BUDGET`, because `_inside` never recorded an own job ID. This supplemental exact-ID accounting audit closes that resource-accounting uncertainty without editing launcher output or manufacturing a successful GPU run.

## Scheduler context and limits

The 05:53:54 UTC scheduler snapshot still shows boron UP / f146 MIXED, six GPUs configured and five allocated, with 16 CPUs and 37120 MiB arithmetically unallocated. This did not establish that this particular request could start. The actual job record reports `Reason=Priority`, requested memory 32G, and also retains `MemPerTres=gpu:40000` / `MinMemoryTRES=40000M`. These fields are preserved as observations; the audit does not claim a unique scheduler root cause, a guaranteed memory-override behavior, or that all physical GPUs were occupied. No scheduling policy or requested resource limit was changed to obtain a result.

## Evidence and reproduction

All native commands were read-only, exact-job or boron scheduler queries with bounded output; all completed with exit 0 and no truncation. `sacct.json`, `scontrol.json`, `squeue.json`, `post-busy-boron.json`, `launcher-files.json` and `launcher-raw.json` include complete argv, UTC and original responses. `audit_accounting.py` independently parses those captures and writes `accounting-proof.json`; it does not call Slurm or execute source. Run it with ordinary Python. The first report-parser failure (slash-bearing scontrol keys) is retained in `audit-parser-negative.txt` and was repaired only in this evidence script.

Field semantics were cross-checked against the primary [sacct manual](https://slurm.schedmd.com/sacct.html): `AllocTRES` describes resources allocated after a job starts, and `ElapsedRaw` is elapsed job time in seconds. The zero-allocation conclusion uses the combined identity, terminal state, no-start, no-node, no-allocated-TRES, zero-elapsed and launcher observations; it does not treat a lone zero utilization counter as proof.

This directory is sealed as `control/final-head-a-postallocation.tar.xz` with a separate per-file SHA256 manifest. The earlier preallocation/source archive is preserved unchanged. B owns the complete broader GPU preparation directory archive; this review retains only the independent allocation/accounting proof and original queue-failure logs.
