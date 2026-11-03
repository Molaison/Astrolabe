# Runtime implementation baseline (2026-11-03)

## Preserved source and scope

The requested plan and review evidence both pin `xcflab/Astrolabe` at
`b537b8a81ff0738922e80a997ffeb9700cdafe09`. The public implementation branch was
`feat/adaptive-runtime-20261102@7a0e2ef43ac15229ab7024ddc88fda00a5adb56a` when
inspected. Work had continued without being pushed. The implementation here
**inherits all commits through `834402daade9f62e9dec0b08041dade163a89369`**, from
the existing `continuation-20261103/integration` checkout on f102. The preserved
commit list is in `evidence/runtime-20261103/preserved-commits.txt`.

That checkout had no tracked or untracked changes reported by
`git status --short --ignore-submodules`. A verified incremental Git bundle was
used to bring its complete commits into a new branch. No existing branch was
reset, rebased, pushed, or merged. The independent checkout based on `261b240`,
including its numerous uncommitted changes, was inspected but not modified or
used as an implicit source for this branch. Selecting that other development
line for final integration remains an explicit repository decision.

The local clone initially contained one untracked resource download temporary
file; it is retained and excluded from the change. Original plan/evidence files
are read-only inputs, not rewritten as implementation evidence.

## Scientific constraints

Execution changes must preserve repeat seeds, request RNG consumption, configured
sample/recycle counts, precision, fixed/shared segments, native joint decoding,
selection and acceptance, and the two existing mutation/stage transaction
boundaries. Unknown execution outcomes must not be retried as new scientific
requests. A daemon process, an import test, and a mocked CUDA call do not certify
weights or residency. Uncertified GPU batching stays disabled.

Existing factories define the target backend set. Mol-1 has already been added
in the preserved commits and therefore is part of this branch; Boltz2 remains a
new variant without an implemented native loader. Protenix exact checkpoint
variants must retain requested and effective identities. No environment or model
upgrades are implicit in this work.

## Available validation resources

The local container has Python 3.12 and no GPU. Its process sandbox rejects Unix
socket creation; broker integration therefore runs in a separate copy on f102
using the existing Python 3.11 licensed scientific container, with all GPUs
hidden. The original source trees and model environments remain unchanged.

Read-only device inventory on f102 showed two RTX 3090 devices with 24 GiB each,
but neither `CUDA_VISIBLE_DEVICES` nor a Slurm allocation was inherited. An idle
device is not an allocation. The GPU identities, CPU/RAM limits, model/weight
paths, and wall-time or GPU-hour budget for real inference must be supplied by
the operator/job scheduler before a GPU benchmark can be claimed. CPU protocol
and scientific replay checks continue independently of that decision.

## Ownership

Three actual concurrent agents own runtime, evaluator, and mutator directories,
respectively. The parent owns CLI/seed integration, benchmark entry points,
cross-component verification and final evidence. Shared files have one writer;
the existing semantic changes are retained rather than reimplemented from main.
