# f102 backend preflight — Agent B, read-only

## Current decision

**ESMFold1 pretrained pilot BLOCKED:** the located folding checkpoint is not a readable ZIP checkpoint. **ProteinMPNN is the smallest concrete GPU-mutator candidate**, with a fully rehashed real checkpoint and an existing complete protein structure. Its scientific protocol and allocation still require root/C freeze; no model inference or GPU use occurred during this preflight. It is not pretrained-evaluator W1 certification.

Device `c429fd25-7aa7-492f-8ff2-d8fa86218e56`, instance `f102-current`, method `command/exec`. Commands use direct argv, timeout 10 s, yield at most 5 s, stdout/stderr cap 4096 bytes. `commands.json` retains commands, native request IDs, complete bounded responses, truncation flags and failures. The initial Python 3.6 subprocess incompatibilities, unavailable `rg`, bounded git-status timeout and explicitly truncated listings remain visible. These were read-only discovery failures, not backend numerical failures.

## Asset and environment evidence

| Component | Exact observed evidence | Qualification |
|---|---|---|
| Existing CPU container entry | `/xcf26/zpzeng/practice_arena/astrolabe_runtime_20261102_implementation/test-container` | Disables CUDA, uses JAX CPU, sets PYTHONDONTWRITEBYTECODE; used only for distribution metadata |
| SIF | `/xcf26/zpzeng/.gh_container/images/GuidedHallucination_af3_licensed_a13_20260722.sif`, 6,418,354,176 bytes | Recorded SHA `242aae3bb97c5c7ea374dab5d7462e04863d9d48ba2ee00cc57e6c0a57a538e3`; fresh `apptainer inspect --json` passed; large SIF was not rehashed |
| Actual interpreter metadata | `/alphafold3_venv/bin/python`, Python 3.11.14; Torch 2.7.1, JAX/JAXLIB 0.4.34, NumPy 1.26.4, AlphaFold3 3.0.1, Protenix 2.0.0, Boltz 2.2.1 | Read without importing models/frameworks or creating CUDA contexts; installed Boltz version is not certification of a Boltz2 registry key |
| ESMFold1 main weight | `/xcf26/zpzeng/.gh_container/assets/esmfold_v1/esmfold_3B_v1.pt`, 2,097,152,000 bytes | Starts `PK0304`, no final ZIP directory, `zipfile.is_zipfile=False`, `BadZipFile`; unusable as located, no repair/download attempted |
| ESM trunk | `/xcf26/zpzeng/.gh_container/cache/runtime/xdg/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt`, 5,678,116,398 bytes; contact regression sibling 6759 bytes | Trunk ZIP central directory readable (75 members), not a full checksum or state-dict validation; fair-esm/OpenFold distribution metadata absent and expected audited ESM source path absent in SIF |
| AF3 | `/xcfhome/pubdata/folding/alphafold3/models/af3.bin`, 1,146,811,260 bytes | Existing restricted pretrained asset recorded ready by historical manifest; not freshly deserialized, rehashed or inferred |
| AF2 | `/pubhome/folding/alphafold/params` | Existing pretrained resources listed in manifest; exact model/variant, features and execution scope still need freezing |
| ProteinMPNN | `/xcf26/zpzeng/gitrepo/ligandmpnn/model_params/proteinmpnn_v_48_020.pt`, 6,681,301 bytes | **Fresh full-file SHA-256** `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd` matches recorded digest; valid ZIP with 120 members |
| Protenix v2 | `/xcf26/zpzeng/.gh_container/assets/Protenix/release_model/protenix-v2.pt`, 1,859,785,497 bytes | Valid ZIP; recorded digest `8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599`; CCD/CIF and RDKit assets exist; full hash/state/config not freshly validated |
| Boltz1 | `/xcf26/zpzeng/.gh_container/assets/boltz_cache/boltz1_conf.ckpt`, 3,595,352,714 bytes; `ccd.pkl`, 345,859,128 bytes | Valid checkpoint ZIP; recorded checkpoint digest `fea245d912c570ec117b2277c2719f312a6fc109c07b6f6ef741690ee775c2f5`; no native model execution |

Historical asset manifest is `/xcf26/zpzeng/.gh_container/assets/software-manifest.json`. Its ready/import/doctor fields are historical records, not new certification. The saved site profile contains old `/xcfhome` binds, `use_nv: true`, `NVIDIA_VISIBLE_DEVICES=all`, and `GH_CUDA_PREINITIALIZE=1`; **it must not be replayed blindly for the shared cluster pilot**. Inherit the approved scheduler allocation instead and bind only the frozen input/source/weights and private outputs.

## Minimal new GPU-mutator candidate for root/C freeze

Use the actual Astrolabe `LigandMPNN` runner with `model_type=protein_mpnn`, or its public `ProteinMPNN` factory. The mathematical implementation is bundled in `astrolabe/mutators/wrappers/ligandmpnn_utils.py`, attributed to upstream `26ec57ac976ade5379920dbd43c7f97a91cf82de`; the observed external native repository has that same HEAD. The local adapters, their modifications, and the exact checkpoint are the identities to freeze; the external repository's full status timed out and is not presumed clean or used as the executed source.

Existing input: `/xcf26/zpzeng/gitrepo/ligandmpnn/inputs/4GYT.pdb`, **531,114 bytes**, SHA-256 `c05c06773ae8c00d2d033f7f7a103c4048981d79418c6c1a2579b15da455b2b7`. PDB ATOM records identify chains A and B with 177 residues each. Preserve the complete 354-residue input, both chain masks/order, actual coordinates and file bytes; do not truncate it to make the budget pass.

Suggested explicit settings for confirmation before execution: `protein_mpnn_v_48_020.pt`, model type `protein_mpnn`, seed 17, temperature 0.3, batch size 1, number_of_batches 1, both chains A/B designed, no fixed/redesigned residue restriction, no symmetry, unchanged bias/omit settings, original native autoregressive distribution semantics. **Freeze every parser/conditioning flag explicitly:** the Astrolabe base YAML sets `parse_atoms_with_zero_occupancy=1`, while the native function default and this CPU readiness probe use False. This preflight is not a selected scientific protocol, and its defaults must not silently replace a baseline setting. Likewise freeze `omit_AA`, bias fields, design patterns, chain order and full decode kwargs. No sampling/precision optimization is proposed. Repeated calls are sequential; compare exact sequences/masks/order and numerical probability arrays with tolerances fixed before measurement, plus RNG/state and observed release. Distinguish cold load, steady requests, full workflow elapsed time, and legitimate completed design requests. Do not label mock scores as pretrained evaluation or claim MCMC acceptance from a standalone decode test.

Root's proposed resource ceiling is **1 exclusively allocated GPU, 4 CPUs, 32 GiB, 15 minutes (0.25 GPU-hours), scheduler immediate wait at most 60 s**. B has not acquired an allocation. Actual executed code path, bounded request count, five interleaved comparison groups and all failed work must be frozen by root/C before GPU use. No output-timing promise is made.

### Follow-up CPU compatibility: DONE, no forward or GPU call

`CPU_COMPATIBILITY.json` and `commands-42-46.json` preserve fresh native import, checkpoint and input compatibility results. Actual imports report Torch **2.7.1+cu126**, CUDA build **12.6**, ProDy **2.6.1**, with `torch.cuda.is_initialized() == False`. The initial new delivery `repo` contained only `.git`, so that first import correctly failed `ModuleNotFoundError`; source was not available yet. The following read-only import used the existing d66 `cloud-20261103-fix` owner. Its two executed MPNN source SHA-256 values exactly match the local current integrated branch (the files have no d66-to-current changes):

- `ligandmpnn_run.py`: `c8aa0cc7ea214a54eeb45f298e3f0e0a9c0fb275b41fc090e6977511286f048f`.
- `ligandmpnn_utils.py`: `8efd500f1791e06c020ee5b87962f225778246a49f2700eda032aac691dc9f75`.

Using read-only asset binds, `torch.load(weights_only=True, map_location='cpu')` loaded the actual pretrained checkpoint. The actual bundled ProteinMPNN architecture accepted **all 118 state tensors strictly, zero missing/unexpected keys**, `num_edges=48`, **1,660,485 float32 parameters**. Native `parse_PDB` with both A/B chains and explicit preflight `parse_atoms_with_zero_occupancy=False` returned **177 + 177 residues**, `X[354,4,3]`, `S[354]`, `mask[354]`. Complete input/checkpoint hashes matched the earlier records. This constructs a model on CPU only to verify state compatibility; it makes **zero forward or sampling calls** and does not warm a GPU model. CUDA remained uninitialized at exit. Old source/worktrees were not edited.

The generic existing endpoint after a real standalone scientific workload has been frozen is `scripts/validation/runtime_workflow_benchmark.py --config <frozen.yaml> --config-dir config --out <fresh-output> --repetitions 5 --walltime-budget <within-approved-900s> --require-gpu`. It requires an inherited GPU allocation. A standalone GPU-mutator harness must identify its narrower scope explicitly rather than pretending to be that complete evaluator workflow.

## Why not silently substitute AF3 here

`repo/config/paper/evaluate_af3.yaml` exists but has unresolved A/B sequences, one seed, five samples and ten recycles. It is not a frozen runtime workload. `examples/benchmarks/af3_inference_benchmark.py` is a separate synthetic-input RefStructure experiment with different sample/recycle defaults and an uncounted compile warmup. Running that script would not reproduce the current runtime contract. GPU compilation within a 15-minute total budget is uncertain; parameters must not be reduced after observing failure to turn a blocked evaluator run into a pass.

## Preservation

`repo@f4b0118`, `backends@95fa925`, and `runtime@db12aa3` were initially observed clean. The separately discovered `cloud-20261103-fix` and `cloud-20261103-opt` contain real uncommitted lifecycle changes; these remain untouched and are archived by C. See `ESMFOLD_EXISTING_WORK_REVIEW.md` for the overlapping existing ESMFold fix.
