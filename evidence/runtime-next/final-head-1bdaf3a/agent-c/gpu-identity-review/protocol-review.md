# ProteinMPNN GPU control: independent scientific protocol review

Agent C, 2026-11-04 UTC. **Protocol recommendation; not a GPU result or a frozen-run certificate.** B owns the harness and workload files; this review does not edit those files. Root freezes their final hashes and source HEAD before any allocated GPU execution.

## Scope and budget

Compare the public `build_runner("ligandmpnn", cfg)` legacy mode and its existing private `ResidentRunner` daemon mode on the **same candidate source**. This tests a real pretrained mutator's fixed-structure decode, serialization and RNG boundary. It is neither a before/after code-speedup experiment nor an evaluator W1 or MCMC trajectory/acceptance test. Direct legacy already holds one native model object across calls; it must not reload per request to manufacture a residency benefit. Shared-broker device admission is outside this private-daemon comparison.

| Item | Required frozen value |
|---|---|
| Input | Full `/xcf26/zpzeng/gitrepo/ligandmpnn/inputs/4GYT.pdb`, 531,114 bytes |
| Input SHA-256 | `c05c06773ae8c00d2d033f7f7a103c4048981d79418c6c1a2579b15da455b2b7` |
| Physical protein positions | A 177 + B 177 = 354, no crop, reorder, synthetic replacement, or omitted chain |
| Checkpoint | Real `proteinmpnn_v_48_020.pt`, 6,681,301 bytes |
| Checkpoint SHA-256 | `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd` |
| Model | `protein_mpnn`, native strict state dict; preflight reports 118 tensors, 1,660,485 float32 parameters, `num_edges=48` |
| Groups | 5, constructor seeds 17, 18, 19, 20, 21 in that order |
| Interleaving | Legacy/resident, resident/legacy, legacy/resident, resident/legacy, legacy/resident |
| Per trial | One model construction, three sequential `design_sequences` calls, then close/release |
| Total fixed budget | 10 model constructions and 30 decode calls, batch size 1 × number of batches 1; no extra forward warmup |
| Scientific stream | Same constructor seed for each pair; no reseed, rollback, or replay between the three calls |
| Temperature | 0.3 |
| Distribution | `legacy_order_conditioned_marginals`, no teacher-forcing substitution or custom decoding ranks |

All checkpoint and SIF/path identities must be literal in the frozen workload/provenance. The preflight counts above are B's CPU-only measurements and need to remain linked to its raw evidence; no GPU claim follows from them.

## Native arguments and input masks

`parse_atoms_with_zero_occupancy=1` is required, matching the existing base setting. `chains_to_design="A,B"` is appropriate because the native design function splits it on commas. **`parse_these_chains_only` must be the list `["A", "B"]`**, not the string `"A,B"`: `_parse_cpu` forwards it unchanged and `parse_PDB` iterates the provided collection. Its native type annotation/docstring describes a list. This was identified from the actual caller and parser; it is not a claim that a GPU run has failed.

Freeze explicit empty values for fixed/redesigned residue lists, symmetry groups/weights, homo-oligomer ties, per-residue and multi-file biases/omissions, membrane-specific selections, design pattern/range, multi-input and supplemental structures. Use additive bias with no bias, batch size 1 and no reference ranks. ProteinMPNN's constructor forces side-chain context to zero for this model; record the effective value rather than implying a different checkpoint protocol. Freeze `return_joint_sequence` explicitly and use the same value in both modes.

**Omission choice accepted by B before any GPU run:** base config has `exclude_AA: [C]`, hence its resolved `omit_AA` is `["C"]`. B initially proposed an empty omission for this separate control, then accepted C's recommendation to use `["C"]`. B also accepted the list-valued parser chain selector and explicit zero-occupancy setting. Root still freezes the final workload bytes. None of these choices was changed after inspecting GPU output or timing.

A CPU-only input parse outside the timed trajectory should preserve full `X[354,4,3]`, native `S[354]`, backbone mask, chain letters/labels, residue indices, insertion codes and ordered chain lists. Hash or store the actual arrays and check the full chain counts/order. Do not silently drop zero-occupancy or incomplete-backbone positions. The original mask determines which positions are valid; a missing backbone coordinate cannot be repaired by redefining the workload after results are seen. All A/B positions are designated for design, and the effective design mask is combined with the native backbone mask. Place parser/import work before scientific seeding or in a separate process, so optional ProDy imports do not consume the trajectory RNG after seeding.

The native legacy branch currently constructs an `out_dict` with masks/native sequence but returns **`output_dict` from `model.sample`**. Actual public output includes `S`, `sampling_probs`, `log_probs`, `logits`, `decoding_order`, distribution semantics, alphabet and chain letters; `return_joint_sequence=True` adds a generated-sequence alias. It does not prove that masks are returned. Record masks as **input-derived parser evidence**, unless additional actual output fields are observed, and do not patch the production return contract merely to make the benchmark easier.

## Scientific gates

| Gate | Pass condition |
|---|---|
| Paired identity | Same source, resolved scientific config, input bytes, checkpoint bytes, device and numerical settings; only mode, trial seed and owned artifact paths vary as declared |
| Discrete result | `S[1,354]`, decoding order, optional generated sequence, alphabet, chain letters, shapes and dtypes match exactly between paired requests |
| Decode completeness | Decoding order is a permutation of all 354 positions; no dropped position/sample or repeated completion masquerading as a new request |
| Floating result | `sampling_probs[1,354,20]`, `log_probs[1,354,21]`, and `logits[1,354,21]` are finite and match with predeclared `rtol=1e-5`, `atol=1e-6`; also report bit equality, maximum absolute/relative difference and offending indices |
| Probability semantics | At effective design positions probabilities are nonnegative and normalize to 1 within a separately frozen numerical check, e.g. absolute 1e-5; mask-zero positions retain native behavior rather than being renormalized by the checker; fixed omission obeyed if applicable |
| RNG | Full Python, NumPy, Torch CPU and actual CUDA generator states, plus Torch numerical settings, match exactly after construction and after each corresponding call; each next request starts from the preceding state, with no intervening reseed |
| True GPU execution | Model parameters and executed tensors actually use the allocated CUDA device; observed lifecycle is weights-loaded + GPU-resident, model load count stays exactly 1 across all three calls |
| Complete outputs | All 30 calls complete once, with original shapes and no NaN/Inf, timeout, retry, truncation or CPU fallback; preserve every raw output and state artifact |
| Release | Close succeeds and the daemon/child process tree and its owned GPU allocations are observed released before the next trial |

Exact sequence/order/RNG failure is a hard scientific failure even when probabilities fall within tolerance. Numerical tolerance cannot be widened after the run. Scores derived from native `S/log_probs` and the frozen effective mask may be compared as a secondary check, but must be identified as independently derived values when the public runner does not return them. Do not assert scientific acceptance/progress: no trajectory optimizer, objective acceptance or stage transition is exercised here.

The control must retain all five paired groups, not a selected subset. A failed trial remains in the record with its configured attempted count and actual valid-complete count. Invalid output must never contribute to the effective-throughput numerator.

## Timing and performance interpretation

Keep request 0 as the first request and 1/2 as the steady requests; there is no excluded forward warmup. Record child cold startup, public runner/model construction, individual synchronized public calls, first-completion latency, CPU-ready output serialization/validation, cleanup, and total spawn-to-exit end-to-end time separately. Constructor duration includes mode-specific identity probing, checkpoint hashing, startup and model initialization; it should not be relabeled as pure model load without instrumentation. Existing resident device-probe processes count toward cold/end-to-end overhead, even though they do not run a model forward.

GPU work must be synchronized before stopping a public-call timer. Canonicalization to CPU and full output writing must have a declared, identical comparison boundary; do not compare a merely enqueued GPU return with a resident CPU-ready result. End-to-end wall time includes the owned process tree and confirmed cleanup. ProcessObserver raw samples should retain CPU/RSS and GPU PID/device/memory evidence so that device fallback or lingering work is observable.

For each group compare `valid_complete / end_to_end_seconds` for the two modes, and separately compare first/steady request latency. Preserve raw elapsed ranges, each paired ratio, the median paired ratio and all failures. A speedup claim requires all scientific gates and complete counts first; five one-device pairs are a limited repeatability observation, not general backend certification. Report zero/negative speedup without changing the benchmark. This comparison cannot satisfy a before/after 20% Runtime improvement target by itself because both modes run the same candidate and the preexisting legacy runner already retains weights.

Root alone selects the fresh Slurm allocation (one GPU, four CPUs, 32 GiB, 15 minutes) after the protocol and source are frozen. No model execution is authorized on an unallocated shared GPU. Allocation expiry or insufficient time makes the fixed run incomplete/BLOCKED; it does not authorize fewer requests or new seeds under the original workload name.

## Review findings sent before implementation

1. Parser chain selector is list-valued; the string split belongs to a different argument.
2. Public result does not include the locally constructed mask/native dictionary; mask evidence must be sourced honestly.
3. Base omission is C; B accepted `["C"]` in place of the initially proposed empty omission before any run.
4. Legacy is already one native model per runner; a per-request reload baseline would be invalid.
5. Constructor/RNG boundaries and public-call return-device differences must be reflected in the checker and timing boundary.

These are protocol/source findings. Actual harness conformance will be reviewed against a fixed file hash once B provides the implementation; actual GPU certification remains pending.

## Offline protocol audit

The workload moved to `benchmarks/runtime/workloads/proteinmpnn_gpu_control.json`. Its reviewed SHA-256 is `6e175fead0abbb39981105053d492050bb654d0744ae06b62a2eb35af5b95c2b`. `audit_protocol.py` independently checks its literal budget/seed/order/settings, both native source hashes, and complete native `design_sequences` signature coverage. It separately parses B's untruncated CPU-SIF command output rather than copying its summary. This audit is **PASS**, with no errors, in `protocol-audit.json`.

The CPU raw confirms A177/B177, mask sum 354 with values `[1]`, CUDA uninitialized, and native sequence SHA-256 `a149f24cd9a0ca21eaeed7478450b91dfec6800b5d372e06655511e6b49a8e44`. The current workload includes the explicit chain-order, mask and native-sequence expectations and probability normalization tolerance. This is a protocol and CPU parse check; it is not independent GPU execution or harness certification.

Reproduce the offline audit from a source checkout and its raw evidence bundle:

```bash
python audit_protocol.py \
  --repo /path/to/candidate-repo \
  --preflight /path/to/f102-backend-preflight/gpu-control-parse-true.json \
  --out /path/to/protocol-audit.json
```
