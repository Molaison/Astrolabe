# Agent B — evaluator/model lifecycle independent investigation

> Historical read-only snapshot at d66. Subsequent implementation, CPU torch availability, independent verification and status changes are documented in `esmfold-output-report.md` and `docs/runtime_backend_certification.md`.

**Scope: initial read-only investigation, 2026-11-04.** No production file was edited and no commit was created. This report selects an implementation opportunity, not a GPU performance result.

## Decision

Preserve the existing process-owned pipeline and resident worker protocols. The smallest evaluator-specific improvement is to split ESMFold1's current `output_to_pdb` into the existing native device computation/host transfer and pure host PDB formatting. A reproducible probe confirms that the current worker retains its device lease during pure CPU formatting and copies an unused model-output field. The probe uses the actual worker's AST and exact pinned vendor helper with fake tensors/model/admission; it establishes the call contract only.

The first global pilot has been selected by the main agent in the producer scheduler, independently of this evaluator recommendation. Agent B will review that pilot before implementing evaluator changes. All true pretrained GPU certifications remain blocked by unavailable allocation, dependencies, and weights.

## 1. Baseline and reuse

- Investigated HEAD: `d66ee714f2e755948534f95d9d125287ce8dbd91`; parent `834402daade9f62e9dec0b08041dade163a89369`; local isolated branch `work/runtime-b-20261104`.
- Production files were clean at investigation start; only this evidence directory is untracked after investigation. Other worktrees and their unmerged changes were not modified.
- Recovered Repository Mental Model and agent context in `../control/Astrolabe_REPOSITORY_MENTAL_MODEL.md` and `../control/Astrolabe_AGENT_CONTEXT.md` describe older `b537b8a81ff0738922e80a997ffeb9700cdafe09`. They were used to locate evaluator factory, stage, and run owners, then refreshed from current source. They are not evidence about current lifecycle behavior.
- Current docs reused: `docs/runtime_implementation_baseline.md`, `docs/runtime_implementation_report.md`, `docs/evaluator_lifecycle_matrix.md`, `docs/mutator_lifecycle_matrix.md`, and `docs/runtime_execution_guide.md`.
- Relevant preserved history: `d66ee71` lifecycle/resource unification; `834402d` Chai component residency; `2637e06` Boltz1 residency; `3289b87` audited AF2 sharing; `d14d75e` / `84c71c1` evaluator evidence provenance; `ec31289` DPLM evidence identity. These are existing mechanisms to strengthen, not rebuild.

Existing real CPU evidence is explicitly negative for resident throughput on its short workload: `evidence/runtime-20261103/cpu-three-mode-02/report.json` reports all 27 trajectories complete, direct median 9.691 s, resident serial 32.418 s, resident adaptive 33.455 s. This does not predict a pretrained GPU workload. Historical snapshot04 reported 403 passed / 1 failed / 1 skipped; the failure was the benchmark child-PID fixture race. Snapshot05's separate 21-test rerun closed it; snapshot06 is a separate 148-test run. Do not merge them into a nonexistent whole-suite run. No historical evidence inspected here certifies true pretrained GPU acceleration.

## 2. Exact registry and certification coverage

`astrolabe/evaluators/utils.py` has **14 registry keys / 13 factories**. `mol1` and `mol-1` are aliases. `boltz` constructs **Boltz1Runner**; Boltz2 is absent. The machine-readable AST inventory records key/factory/source position and source SHA-256 in `registry-inventory.json`. AST inventory does not import or certify any backend.

Status notation: **contract only** = mock, tiny CPU model, or factory/runtime contract evidence; **U** = real pretrained model and real GPU workload unverified; **N/A** = CPU-only target. Rows separate known variants but cannot certify arbitrary configured checkpoints or source versions. Every real run must record actual source, all weight/assets, environment, effective configuration, and physical device identity.

| Exact registry key / variant | Identity needed for certification | Existing evidence strength | True execution status |
|---|---|---|---|
| `alphafold3` | vendor `4b154d922759c918dd4b5a4d5a779853100fd20e`; full actual model_dir digest; native model config, seeds/recycles/samples/precision, topology/CCD | Existing split stages, identity and content-cache contracts; no pretrained run | U |
| `alphafold2`, default model_3_multimer_v3 | vendor `c77e5d2a8961d1a353632c462914ff0a32a950f6`; `params_model_{model_id}_{suffix}.npz`; effective config; audited predict digest for sharing | Existing tiny CPU runner, explicit-seed/fixed-params/shared-worker contracts | U |
| `alphafold2`, other IDs / ptm etc. | Each actual model_id/suffix/checkpoint/config is a separate scope | Factory support is not variant certification | U |
| `rf3` | Actual Foundry `src/`, `models/rf3/src/`, ckpt_path and Python environment; effective config including seed | Common worker, engine identity, post-init RNG replay contracts | U |
| `esmfold` (ESMFold1) | `esmfold_3B_v1.pt` or exact local .pt; ESM/esm2 `43c67d952da30d2471f609848931d20ec8c98532`; OpenFold `8396098268a5675b0ff776ec274f15d380a4ccf5`; actual hub dependencies and torch settings | Existing constructor/RNG/continuous-request contracts; new exact-source boundary probe | U |
| `esmfold2` (`biohub/ESMFold2`) | ESM `ba4d7124864eed323a93bf3cfefcd958f573b75a`; Transformers `ef32577f55da19a4989cd7b22e004dc43a4998cb`; both local folding and ESMC snapshots; loops/diffusion/sampling/precision/seed | Existing immutable settings, dual-loader, local-files-only, metadata contracts | U |
| `protenix`: protenix_base_default_v0.5.0 | Actual v0.5 default checkpoint/source/env; effective config | Tiny CPU RNG/dynamic skip_amp and factory identity | U |
| `protenix`: protenix_base_constraint_v0.5.0 | Dedicated constraint checkpoint/source/config; strict vs compat requested/effective model | Factory identity and constraint handling contracts | U |
| `protenix`: protenix_base_default_v1.0.0 | Exact v1 source/checkpoint/environment | Factory identity only is not v1 model validation | U |
| `protenix`: protenix-v2 | Reference vendor `2475421477ab414b571149ad4a875c390ff8a35d`; actual source/checkpoint/env; dtype/skip_amp/mc_dropout | Generic adapter tests do not certify every model version | U |
| `opendde` | vendor `266ce4c49d492ad1077866000d83704999985f46`; actual model_name/checkpoint/CCD/templates; single-device single FoldCP | Common worker, preprocessing/native composite/RNG contracts | U |
| `mol-1`, `mol1` aliases | Actual `mol/` source; default mol-1-base or configured checkpoint; CCD/cluster/obsolete assets | Independent adapter contracts, protein-only factory input; daemon execution required | U |
| `boltz` (Boltz1) | vendor `cb04aeccdd480fd4db707f0bbafde538397fa2ac`; actual boltz1_conf.ckpt; Lightning/env and effective settings | Lazy load count (hello0, first request1), precision/RNG contracts | U |
| `chai1`, default CPU component offload | vendor `c544fb183e865c4950909444db860a9d50604f66` or audited local variant; six component checkpoints, optional ESM2 JIT, conformers | Component cache/RNG/lease contracts | U |
| `chai1`, component_residency=device | Same complete identity plus explicit residency config; allocation must fit simultaneous components | Configuration support is not speed/VRAM certification | U |
| `rhofold`, unrelaxed | vendor `6bdfbda720184409eb682ce08c05d258162ddc48`; checkpoint + local RNA-FM dependencies; config | Tiny CPU nn.Module, two requests/one load, sigmoid/softmax/RNG contracts | U |
| `rhofold`, OpenMM relaxation | Above plus OpenMM/platform/env and verified physical device binding | CPU physical-binding acceptance/rejection contracts only | U |
| `dummy` | Repository revision and complete deterministic config; PyRosetta if writing structures | Historical actual CPU trajectory E2E and three-mode benchmark | CPU validated for recorded workload; GPU N/A |
| `rosetta_crystal` | Actual PyRosetta environment, .params, reference structure, repack/minimize/seed/replicates/mapping settings | Fake-reference and invalidation contracts | True PyRosetta run unverified; GPU N/A |

Existing per-family evidence is indexed in `docs/evaluator_lifecycle_matrix.md` section 6 and its named test files. `registry-inventory.json` is a current extraction; this table is a scope/evidence audit, not a new claim that those historical tests ran in this environment.

## 3. Actual call graph and resource owners

Top-level caller: registry/factory → evaluator `prepare_input/run_function/prepare_output`. Existing resident clients own request artifacts and CPU work; broker-managed model workers own expensive models and device execution. The trajectory owner hides GPUs; only the configured worker gets a binding. Existing `operation(..., phase='device')` surrounds the nested admitted GPU segment. The opportunity is to shorten specific composite segments without changing their mathematical work.

| Family / owner | Before nested device lease | Current device-lease segment | After lease / remaining boundary |
|---|---|---|---|
| AF3 wrapper/resident | Owner featurization/cache; worker decode | Native model forward and D2H | Tree codec, owner structure extraction/scoring |
| AF2 wrapper/resident | Owner prepare_only/process_features; worker decode | RunModel.predict with explicit seed and D2H | Codec and owner result extraction/scoring |
| Protenix wrapper/resident | CPU feature preparation | Restore RNG, update config, predict/dynamic skip_amp, D2H, capture RNG | Codec and CPU dumper |
| ESMFold1 daemon | Decode request | `model.infer`; full native `output_to_pdb` including CPU formatting; score D2H; RNG | Codec only; PDB formatting demonstrably still inside |
| ESMFold2 worker | CPU job parsing/preparation | Native builder.fold + complete result materialization (composite) | File writing; finer native split not independently established |
| OpenDDE / Mol1 workers | Native CPU preprocess | Native inference loop/dataloader/model/output/cleanup (composite) | Protocol response; native loop needs dedicated audit |
| Chai worker | CPU feature context | ESM segment separately; nativefold including CPU ranking/file composite | External preparation/finalization only where proven |
| Boltz1 worker | CPU YAML/MSA | Lightning predict + dataloader/transfer/writer (composite) | External request protocol |
| RF3 worker | Request/config handling | `engine.run` native pipeline/dump composite | Protocol response |
| RhoFold worker | Owner tokens; worker decode | Model/RNA-FM; original sigmoid/softmax; D2H | CPU NPZ/CT/PDB; optional OpenMM separate admitted composite |
| Dummy / Rosetta | CPU owner only | No GPU execution | CPU scoring/output/close |

Composite segments are already documented as limitations, not newly introduced regressions. Moving their interior work needs backend-specific proof; a method name such as `finalize_host` alone is insufficient.

## 4. Reproduced ESMFold1 boundary defect

Source owner: `astrolabe/evaluators/wrappers/esmfold_daemon.py::main.handle`. At investigated HEAD it calls `model.output_to_pdb(output)` inside `operation(actual, phase='device')`, then separately copies selected scores through `cpu_tree`.

Exact pinned native source was fetched from [Molaison/gh-esm2 at 43c67d9, esm2/esmfold/v1/misc.py](https://github.com/Molaison/gh-esm2/blob/43c67d952da30d2471f609848931d20ec8c98532/esm2/esmfold/v1/misc.py), blob `76b8e74cd73d44f68ea197fd15d5e2d38cbf8284`. The saved diagnostic copy and source receipt are `esmfold1-native-misc.py` / `esmfold1-native-source.json`.

Native execution order: atom14_to_atom37 on the output device → **all** output tensors `.to('cpu').numpy()` → atom positions to CPU → loop over batch, build OFProtein, `to_pdb`. The final part is pure host NumPy/string formatting. Copying the entire output also includes fields that neither PDB nor selected scores need. The current worker copies selected fields again afterward. This establishes redundant transfer opportunities; their real cost remains unmeasured.

Reproduction from this worktree:

```bash
../venv/bin/python evidence/runtime-next/agent-b/lease_boundary_probe.py
../venv/bin/python evidence/runtime-next/agent-b/lease_boundary_probe.py --require-host-finalization
```

The first command exits **0** and reports `host_formatter_inside_lease=true`, `unused_output_copied=true`. The second command is a deliberately failing gate and exits **1** on this baseline. Both runs are retained in `probe-replay.json`, with full event streams in `lease-boundary-probe.json` and `lease-boundary-failing-gate.json`.

Evidence scope: real worker AST + exact pinned helper; fake tensors, model, formatter and admission. **No torch kernels, pretrained weights, physical GPU, real broker transport, or timing are exercised.** This is a reproducible code-contract finding, not measured GPU speedup.

### Architect alternatives

Caller-first API is the existing `resident.run(ordered_sequences)`; callers continue receiving scores, PDBs, and RNG state with unchanged order and schema.

| Design | Model/data ownership | Benefit | Risk / decision |
|---|---|---|---|
| A: ESMFold-specific audited two-stage helper in the worker | Model stays worker-owned; same native atom14→37 on device, copy only exact required host fields/scores while admitted; release lease; same native OFProtein/to_pdb on NumPy host payload | Shorter lease; removes unused output transfer; preserves transport/caller API | **Select for evaluator phase.** Audit actual helper implementation; fallback to existing composite for unsupported/custom variants. Differential PDB/arrays/RNG required |
| B: Transfer full inference output to owner, invoke native output_to_pdb on CPU | Full outputs cross worker/owner boundary; atom remapping moves device to CPU unless extra staging added | Simpler apparent host ownership | Reject: adds IPC and full-output copy, changes compute placement/possible dtype behavior, exposes internal tensors, broadens protocol. No performance/science evidence |
| C: Add generic lifecycle DAG/actor framework | New global scheduling and lifecycle abstractions | Broad future flexibility | Reject for this iteration: duplicates existing process-owned pipeline, greatly increases migration surface, no workload evidence justifies it |

A safe implementation must preserve atom order, masks, `residue_index + 1`, chain_index behavior, b-factors, PDB byte content, selected score arrays, sample/recycle/precision settings and exact RNG consumption. Device work must be completed before releasing its lease; `.cpu()`/NumPy materialization must remain inside where it is the synchronization boundary. `infer` itself has CPU preprocessing; separating that is explicitly out of this minimal change.

Required independent verification: run actual helper with CPU torch tensors and the pinned native functions, compare byte-identical PDB plus exact selected arrays; test multichain, absent chain_index, batch size >1, dtype/device behavior and unsupported helper fallback. Retain actual pretrained GPU runs as separate gates.

## 5. Incremental preprocessing and strict invalidation

`astrolabe/feature_cache.py` already requires exactly 15 dependencies: schema, backend, source, weights, dtype, mapping, chains, chain_order, topology, constraints, seed, msa, templates, ccd, options. Missing/extra fields are rejected; canonical JSON disallows NaN and hashes with SHA-256. This existing mechanism must be reused.

AF3 `astrolabe/evaluators/alphafold3.py::run_function` validates resident config before cache access. Its opt-in, single-slot feature cache removes the irrelevant request `name`, but includes all sequence/chain/topology dependencies, seed list, MSA/templates, CCD, full preparation options and initial-guess file content digest. Hits and stored examples are deep-copied. Sequence changes invalidate the full feature set. Unsupported `daemon_constraints` are rejected; supported cyclic/jump options are explicit. The wrapper's prepare/forward/finalize boundary is already real; do not replace it with new generic caching.

Source/CCD static identity is cached for a wrapper lifetime. Existing contract requires source/assets to remain frozen for that lifetime; this is not evidence of an arbitrary safe hot-reload. The single slot can thrash when alternating identities, but no current profile quantifies this. Chain-level or LRU caching is deferred until a real workload establishes repeated eligible keys and strict dependency equivalence.

New local low-cost replay:

```bash
../venv/bin/python -m pytest tests/test_backend_stages_cpu.py -q -k 'not af3'
```

Result **3 passed, 1 deselected, 0.08s**, retained in `cache-contract-tests.txt`. These cover the selected content/identity/constraint contracts; the deselected AF3 path needs vendor dependencies. This is not AF3 model validation.

## 6. Genuine GPU certification entry and blockers

Current blockers: no GPU/device allocation, no pretrained checkpoint assets, no torch/vendor environments, no `/xcf26` mount/SSH destination; local AF_UNIX and process-namespace restrictions prevent genuine broker E2E. No attempt was made to evade these restrictions or claim an idle/shared GPU as permission. CPU genuine-transport CI is being prepared by Agent A; at report time no actual CI run exists for this iteration.

A repository-supported real smoke entry is `config/container_af2_smoke.yaml` (A20, model3/multimer_v3, recycle1, steps2, mutation_num0, no PyRosetta structures). AF3 smoke is `config/container_af3_smoke.yaml` (A20, seed1, recycle1, sample1, steps3, pipeline disabled, cudnn). These are existing smoke configurations, not substitutes for a frozen representative workload.

Inside an **already authorized allocation**, after installing pinned dependencies and supplying the configuration's real local weights:

```bash
python -m astrolabe --config-dir config run --standalone /absolute/path/frozen-workload.yaml --no-chdir
python scripts/validation/runtime_workflow_benchmark.py   --config /absolute/path/frozen-workload.yaml --config-dir config   --out /absolute/path/new-empty-benchmark-directory   --repetitions 3 --walltime-budget APPROVED_SECONDS --require-gpu
```

`APPROVED_SECONDS` must be replaced by the actual allocated budget; `CUDA_VISIBLE_DEVICES` must be inherited from that allocation. The benchmark script supports direct, resident serial and resident adaptive, interleaved orders, science fingerprints and effective completion counts. `--direct-repo` can select a frozen earlier checkout. `--plan-only` creates a plan but is not a run. Do not label AF2 smoke throughput as representative ESMFold1 benefit.

For ESMFold1, root must freeze a real supported workload/checkpoint before performance testing. Required gates are direct vs resident numerical/PDB/RNG/selection equivalence; single + at least two continuous requests with one load; invalidation and cross-request isolation; process/VRAM recovery after failure; >=3 interleaved repeated fixed-workload runs. Report effective complete jobs per wall second, cold/warm cost, CPU/RAM/VRAM and failures. Only true hardware runs may establish reproducible GPU throughput improvement.

## Delivery status

| Item | Status | Evidence / remaining gate |
|---|---|---|
| HEAD/worktree/history/old Mental Model delta | DONE | Read-only baseline above; no production edits |
| Exact registry and variant evidence audit | DONE as audit | 14 keys/13 factories, matrix; every pretrained scope U |
| ESMFold lease defect and redundant-copy diagnostic | DONE as code contract | Baseline gate exits1; raw events + exact native source retained |
| Minimum architecture recommendation | DONE | A selected for subsequent evaluator phase; main pilot remains producer identity |
| Local cache/identity selected tests | DONE in recorded scope | 3 passed / 1 deselected |
| Evaluator production implementation/differential CPU tensor tests | PARTIAL / not yet authorized phase | Wait for first pilot and independent review; no code changed |
| True pretrained GPU/backend matrix certification and speed claim | BLOCKED | Allocation, weights, complete runtime environments unavailable |
| User-specified server output destination | BLOCKED in this environment | No mount/SSH; local isolated worktree evidence handed to root |
