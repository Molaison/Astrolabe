# Independent review: ProteinMPNN GPU control harness

**PASS for the frozen scientific protocol and CPU checker contract at `4889f7b7c08828a99b729719f0c668eaaf4d2396`. Actual GPU execution remains unperformed by this review.** Agent C reviewed and tested an independent detached checkout. The first candidate `1b7a1bdc32dda273fac6b4e7bebad8a5a9c1f0ca` had a reproducible child-identity acceptance gap; the revised candidate closes that gap without changing the native model, scientific parameters or workload budget.

| Exact artifact | SHA-256 |
|---|---|
| Reviewed harness `benchmarks/runtime/gpu_mutator_control.py` | `0c789ab58f2e9c044da84e5718a647eb099fda4b306026ef694ea301e6f72aa6` |
| Reviewed workload `benchmarks/runtime/workloads/proteinmpnn_gpu_control.json` | `72a18eef550fdf5555501b4163126522d752f441c3b7a5aad517a852d6498fc6` |
| First harness with the acceptance gap | `e1c3f56df4ea3b20c2da67f30f627f56688b97a946ef66368eaeaf4cade65ad4` |

## What was independently checked

The native caller/parser/output source was read directly. Every decode argument is explicit and matches the native signature. The protocol retains the real checkpoint and entire 4GYT A/B 354-position structure, all-one backbone masks, zero-occupancy parsing, C omission, temperature 0.3, batch 1 × one batch, no symmetry and native legacy-order-conditioned sampling. Seeds 17–21 define five interleaved mode pairs; each trial constructs one runner and makes three ordered calls without reseeding or warmup. Ten GPU model constructions and thirty native decode calls are planned. The separate CPU preflight construction and resident device-identity probes are explicitly accounted for.

Source and literal protocol checks are independently reproduced by `audit_protocol.py`. It reads the actual CPU-native preflight stdout, not the author's conclusion. `protocol-audit-4889f7b.json` is PASS and confirms native source hashes, full decode-signature coverage, budgets, chain counts, masks, native sequence hash and numerical settings. It remains CPU parse/source evidence. Earlier `protocol-audit.json` records the preceding protocol hash before output/telemetry-budget fields were added; the scientific inputs and decode parameters were unchanged.

The harness calls the public runner construction, `design_sequences` and `close` methods. Legacy is already a resident model object and is not reloaded on each call. Each public-call timer includes CPU materialization through the existing `cpu_tree` helper and CUDA synchronization. Constructor timing includes private-runner startup/identity work, rather than claiming a pure checkpoint-load measurement. The primary `run_once` wall time spans fresh process creation, imports, asset/source verification, construction, three requests, output recording and confirmed observed release. Root's Slurm accounting must additionally cover full allocation startup/shutdown.

The native public result does not return its locally constructed masks/native dictionary. The harness correctly records masks as independent CPU input-parse evidence and validates the real returned S/probability/log-probability/logit/order/chain/alphabet fields. Discrete results and full CPU/Python/NumPy/CUDA states are exact between modes. Consecutive request states must be continuous and CUDA RNG must advance. Floating outputs use the predeclared tolerances and preserve bit equality and numerical deltas.

Missing or failed work, forced cleanup, nonzero exit, incomplete release, wrong shape/sequence/probabilities/RNG or mismatching pair stops later groups. Invalid comparisons retain raw attempts and diagnostic counts but omit certified effective throughput. The controller uses the existing bounded process observer/telemetry harness and output budget; no separate process ownership mechanism was introduced. Missing VRAM/RSS observations remain explicitly unknown and do not become zero measurements.

## Reproduced failure and repair

C's independent fault injection used the author's explicitly synthetic CPU fixture only as a valid starting record, then changed each child field independently. It never ran a model or GPU, and none of its timings is a benchmark.

| Injected child error | First candidate `1b7a1bd` | Revised `4889f7b` |
|---|---|---|
| Seed becomes 999 | Incorrect PASS; effective throughput present | INVALID; effective throughput absent |
| Resident child reports legacy mode | Incorrect PASS | INVALID |
| Foreign source HEAD | Incorrect PASS | INVALID |
| All request indices become 0 | Incorrect PASS | INVALID |
| All request phases become first | Incorrect PASS | INVALID |
| Unmodified positive control | PASS | PASS |

The first candidate validated the ordered controller row prefix but ignored the child identity and request ordering in `trial.json`. The fix requires every child plan field with its exact type/value, the frozen source HEAD and manifest hash, request indices 0/1/2, and phases first/steady/steady. The author also added finite positive phase timing checks bounded by total observed wall time. The protocol SHA did not change between these commits.

Raw evidence is `child-identity-before.json` (initial combined mutation), `child-identity-before-separate.json` (five faults on an exact detached first-candidate checkout), and `child-identity-after-4889f7b.json` (same five faults after repair). `probe_child_identity.py` reruns these checks against either checkout.

## Tests and reproducibility

Independent focused execution on `1b7a1bd`: **52 passed**, with the uncovered identity fault separately demonstrated. Independent focused execution on the fixed detached `4889f7b`: **62 passed, zero skips**. Logs/JUnit are `b-1b7a1bd-tests.*` and `b-4889f7b-tests.*`; the latter run took 13.46 seconds in this local CPU environment. These durations describe tests, not model speed.

```bash
python -m pytest -q tests/test_gpu_mutator_control.py
python /path/to/probe_child_identity.py --repo /path/to/4889-checkout --out /path/to/child-identity.json
python /path/to/audit_protocol.py --repo /path/to/4889-checkout \
  --preflight /path/to/f102-backend-preflight/gpu-control-parse-true.json \
  --out /path/to/protocol-audit.json
```

The first command runs from the source checkout. Use distinct new output locations when repeating evidence collection. The independent reviewer did not modify B's source checkout or run GPU inference.

## Certification boundary

Root can now freeze the combined candidate and exact manifests before its separately authorized allocation. This review does not certify actual CUDA execution, hardware resource release, floating equality on the device, performance or pretrained evaluator/MCMC behavior. Those require the full run and independent raw-data audit. Any later harness, protocol or native source change requires its relevant gate to be rechecked.

The eventual experiment is a **same-candidate mode comparison** of one ProteinMPNN model. It does not attribute speed to a newly introduced residency mechanism, certify other checkpoints/model types, prove MCMC acceptance or stage correctness, or replace W0's fixed-revision comparison and original performance target. Both modes already retain weights; a slower resident result must remain in the report.

## Separate A Runtime scientific-boundary probe

`probe_reused_runtime.py` independently checked exact A source `60684bd4fefa6e9b7282176e8b5af7700ef98093` with real State/ledger/cancellation and a controlled worker boundary, explicitly substituting only post-response process observations. All three cases passed in `reused-runtime-60684bd.json`: outcome-unknown dispatched once and could not be replayed while another CPU worker remained usable; a wrong-owner cancel had no effect and a certified queued rejection allowed only an explicit later retry; a cancelled unsubmitted staged execute preserved the same prepared owner/token, with zero dispatch until explicit resume and finalize. These are ownership probes, not real IPC/GPU or throughput evidence. Root's separate capable-host full suite remains the execution authority for the broader Runtime integration.
