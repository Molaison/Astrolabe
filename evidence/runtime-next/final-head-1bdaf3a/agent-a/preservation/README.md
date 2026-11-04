# Independent preservation gate for original unmerged work

**PASS_PRESERVED — no observed differences.** Agent A performed three bounded read-only native snapshots on 2026-11-04 06:02:40–43 UTC and compared them with the original Agent C snapshots in `control/f102-source-preflight`.

| Original linked worktree | HEAD / branch / Git topology / status | Original unmerged files | Hash scope |
|---|---|---:|---:|
| `cloud-20261103-review` | Unchanged; d66 and clean | 0 | 12 paths: 11 files + 1 previously absent helper |
| `cloud-20261103-fix` | Unchanged; d66, original fix branch | 14 tracked + 7 untracked, all preserved | 30 paths: 29 files + 1 previously absent helper |
| `cloud-20261103-opt` | Unchanged; d66, original opt branch | 16 tracked + 9 untracked, all preserved | 33 paths: 32 files + 1 previously absent helper |

The absolute paths and common Git directory match the initial topology: these are linked worktrees rooted under `/xcf26/zpzeng/practice_arena/astrolabe_runtime_20261102_implementation`, sharing the old `cloud-20261103-test/.git` common directory. No other worktree was checked out, reset, restored or edited.

Every originally modified tracked file and every originally untracked source/test file was individually SHA256-compared. The complete original tracked patch digests also match: fix `49c84419808453d6629cf3f3464305590ea57f8a38e7b9655358ca61d1904f56`; opt `9f14f4fea5634d64e3a4aa40a3f851b9d52014475800a61b660f7c91dc50f2bd`. The original relevant-owner and report hashes were checked additionally. The three original absent `esmfold_output.py` paths remain absent; our new helper was not copied into the older worktrees.

The audit hashed 72 existing files totaling 1,611,462 bytes and checked three expected absences. It did not scan model weights, large resource files, ignored artifacts or submodule contents. The source/test content hashes and normal-mode status are sufficient for the recorded unmerged-change preservation gate; they are not a claim that every ignored file or any transient write between observations was monitored.

## Provenance and reproduction

`expected.json` records original paths, metadata and hashes. For tracked files that initially existed only as a preserved diff, `build_expected.py` reconstructs their exact postimages **in local memory** from the d66 Git blob and original patch. It validates all old/context lines, every hunk count, both old/new Git blob prefixes, the full original patch digest and overlapping originally recorded SHA256 values. It never applies a patch to a worktree.

`probe_remote.py` is the exact Python 3.6-compatible read-only native probe. All three requests set `GIT_OPTIONAL_LOCKS=0` and `PYTHONDONTWRITEBYTECODE=1`; all exited 0 without truncation. Git diff content was restricted to the originally modified tracked paths, and SHA reads to explicit original/owner/report files (hard 2 MB per-file cap). Compressed responses were SHA256-verified before decoding. `*-native.json` retains the complete request argv and response; `*-current.json` contains the losslessly decoded records.

`compare.py` compares these captures offline and produces `preservation-proof.json`, containing all before/current file checks. It reports any difference with owner UNKNOWN and does not repair anything. Its current `differences` list is empty. Re-run `python compare.py` from this extracted directory to reproduce the comparison; original expectation reconstruction additionally needs the existing initial evidence and local d66 Git objects.

`negative-parser-note.txt` preserves a small local report-parser issue: the original bounded response's unrelated old-test tail was truncated. The complete first review record was decoded independently; no incomplete old-test data was used to establish this gate. No remote mutation, scientific evaluation, GPU query or benchmark was performed.
