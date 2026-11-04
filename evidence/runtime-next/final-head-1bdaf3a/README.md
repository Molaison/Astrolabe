# Final source verification — 1bdaf3a

Production HEAD: **1bdaf3a53227c9197b930dc00a01b752acd90067**.
Tree: **6f763bfb04778ef4f32386ddb8aeb38ce4240bf1**.
The source checkout and PR #9 stay at this tested HEAD; this branch adds only documentation and evidence.

## Outcome

| Scope | Status | Original evidence |
|---|---|---|
| Exact final-HEAD CPU and geometry contracts | DONE | [C](agent-c/README.md): 378 +181 tests, zero skips/failures; raw GitHub ZIPs, every extracted artifact, command/environment/JUnit and source manifests |
| Exact final-HEAD full W0 | DONE | 30 trials /90 trajectories /270 evaluations; C's complete independent reader and [lead direct raw audit](root/raw-independent-audit.json) |
| Exact final-HEAD scientific positive control | DONE | Actual d66/final CLI runs: 4 evaluations and 5 rows each, one accepted mutation, one stage transition; full Python/NumPy/Torch CPU RNG and terminal state exact |
| Production equivalence to measured f9 | DONE | [A](agent-a/README.md): 948 paths, 917 blob bytes/SHA and 31 gitlink references; four measurement modules exactly d66/f9/final |
| Original unmerged work preservation | DONE for recorded scope | [Final preservation proof](agent-a/preservation/README.md): all three old worktrees retain original HEAD/branch/status/topology; fix 14+7 and opt 16+9 recorded changed/new files and patch hashes unchanged |
| Real pretrained checkpoint/input CPU freeze | DONE | [B GPU supplement](../../../docs/runtime_gpu_mutator_certification.md): real 1,660,485 float32 parameters, full 354-residue 4GYT input, zero forward/sample/CUDA initialization |
| Pretrained GPU control | BLOCKED | One bounded Slurm request, job 33334; queued 60 s, nodes busy, no allocation/inference. Complete launch/failure evidence and A's independent terminal accounting retained |
| Actual allocation cost/release | DONE for the unallocated request | CANCELLED, Start=None, ElapsedRaw=0, AllocCPUS=0, AllocTRES empty, own squeue empty; **0 allocated GPU-hours**, no GPU execution/release certificate |
| Wider Runtime backlog | PARTIAL | [Final report and R0–R12 boundaries](../../../docs/runtime_final_report.md); no durable exactly-once, complete async producer, evaluator W1 or full multi-GPU certification |

The final CI's aggregate CPU eval/s gains versus d66 are +63.81% /+84.33% /+86.69% for direct/serial/adaptive. Each mode has five faster pairs and separated observed ranges. The distinct f102/SIF experiment compares 9d to f9 and yields +69.08% /+69.99% /+54.92%. These comparisons use different baselines/hosts; their rates are not combined. Resident remains slower than the same-version direct path on this short workload.

W0 includes 90 accepted initialization rows and 180 rejected mutation attempts, with no accepted mutation progress or structure output. Its separate positive control covers acceptance/stage behavior. Neither CPU experiment certifies GPU science or speedup.

## Independent verification and provenance

- [Agent C archive and readers](agent-c/README.md) preserve every actual final-CI and positive-control record. The server ZIP digests and every extracted file are independently checked.
- [Agent A source/resource review](agent-a/README.md) separates observed production equivalence, pre-allocation scheduling snapshots and exact own-job terminal accounting.
- [Lead raw audit](root/audit_final_head_root.py) reads every original JUnit/CSV/stage stream directly, recalculates timing, checks complete terminal scientific JSON and records input hashes. It does not invoke C's reader or perform new inference.
- [Publication mapping](root/published-all.json) maps 28 original continuation commits to published commits with identical trees. The original source anchors remain in the earlier bundles.
- [Final source bundle manifest](root/final-source-1bdaf3a-manifest-30k.json) and [delivery receipt](root/native-results.json) prove the actual server checkout HEAD/tree/clean state. A transient connector outage and rejected oversized transfer are retained; neither started a job or repeated a mutation.

Raw evidence remains unchanged. Archive SHA256 entries are also listed in the parent `archives.sha256`; run that check from `evidence/runtime-next`. Each owner's README gives safe extraction and recheck commands. Source-affecting files do not change on this evidence branch, so the final runtime HEAD remains the one used in CI and native freeze.

After extracting C's raw archive, the lead's separate recheck accepts explicit paths:

```bash
python root/audit_final_head_root.py \
  --raw /absolute/path/to/extracted/final-head-c \
  --output /absolute/path/to/new-root-review
```

This reader uses Python's standard library; it does not require GPU access or rerun the scientific workload. Consult C's archive manifest for its extraction layout before selecting `--raw`.

## Real destination

`/xcf26/zpzeng/practice_arena/astrolabe_runtime_20261102_implementation/cloud-20261104`

`source-final-1bdaf3a/` is the exact immutable source; the separate final evidence checkout contains this report and all historical/current raw archives. GPU failure directories and original earlier worktrees remain preserved.
