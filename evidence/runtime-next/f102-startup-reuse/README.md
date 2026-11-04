# Reused native startup optimization: candidate and bounded evidence

Date: 2026-11-04. Author/owner: Agent A. Independent native reviewer: Agent C.

**DONE:** reuse of the recovered old implementation, local red/green and registry
guards, and four bounded real-SIF cold diagnostics. **PARTIAL:** performance KEEP
is pending root's full frozen W0 comparison. No end-to-end speedup is claimed by
this report. This candidate has not changed a scientific workload or GPU backend.

## Source and ownership

| Item | Exact revision / provenance |
|---|---|
| Existing integrated baseline | `47c6f6dd18fe822a6d28ea6970c756d5dfae7b95`, tree `f63f4e680d946cc30fd3981fe31fbf5aa4be4d12` |
| Recovered tests only | `3852daa`: two original old-opt test files copied byte-for-byte |
| Performance candidate | `f9ce87b9f5cf37710f0a52ebe503595e5d7fdd7f`, tree `00cde2af9ed9c2a4778cf1e4e82e5a9e18183aba` |
| Original implementation | C's SHA-verified read-only snapshot of `cloud-20261103-opt`, based on d66 |
| A isolated local worktree | `agent-a-f102-startup`, created from the exact integrated baseline |

Only two production owners change. `astrolabe/runtime_requirements.py` is the exact
old-opt file (SHA-256 `c60dcfa62fcd5ea39246bff5373ebc548bfc106ac9b12d2d242b8da676a6d469`).
`astrolabe/losses/interaction.py` combines the recovered PyRosetta lazy loader with
the already validated SciPy lazy imports. It does not replace the current file
with the older eager-SciPy version. The resulting SHA-256 is
`a78914c2a6b5e410b9f07935ead280f15f9d714488c3655d8cd958a5febb322c`.

The loader publishes the existing seven native symbols only after a complete
successful import under its existing recovered lock. It caches original import
failures and keeps the prior ImportError message/cause. It does not call native
initialization. Six helper bodies gain the recovered initial guard; their remaining
ASTs are identical. The other 136 interaction functions are unchanged. All registry
decorators and signatures remain present. No scheduler, broker, guardian, model,
seed, initialization-option, scientific-configuration, or frozen measurement module
is modified. The remote old worktrees and root's main worktree remain untouched.

This is a new environment-specific follow-up to R1, not a reversal of the earlier
minimal-CPU R1-round-2 STOP. The real f102 SIF has installed PyRosetta; the earlier
minimal CPU environment did not. Recovering the already written old-opt mechanism
avoids rebuilding it and exposes a previously unavailable cost to actual measurement.

## Caller-first hypothesis

`Astrolabe.run` loads `run.py`, whose loss registry discovery imports interaction.
The baseline eagerly imports installed PyRosetta even for sequence-only W0. The
candidate defers only that import until an existing native-loss guard is reached.

`parse_arguments` persists each run configuration through
`build_persisted_run_config` → `_prune_env_sections` → `task_requirements` →
`_pyrosetta_loss_types_from_source`. The scanner's old implementation calls
`ast.get_source_segment` for every registered function. The recovered version
splits each source once into UTF-8 lines and applies the parser's byte offsets.
It retains the parsed tree, registration tests, dependency heuristics, returned
set, and one-entry lru_cache. This work occurs in real configuration persistence,
not only in the doctor command.

## Local correctness and preserved red results

| Gate | Before | Candidate | Meaning |
|---|---:|---:|---|
| Original lazy-native tests | 8 failed | 8 passed | Before failures explicitly observe the eager import; tests use isolated fake native symbols and never initialize real PyRosetta |
| Original source-slice tests | Collection error: helper absent | 10 passed | UTF-8, CR/LF, embedded Unicode separators, and all registered source functions preserve the exact old AST source text |
| Existing cold SciPy/RNG and registry suites | Existing validated contract | 4 passed | The earlier spatial lazy-import behavior is retained |
| Independent before/after snapshot | 169 entries | Exactly identical | All metadata, modules, signatures, all 169 per-loss requirement results, and 40 inferred PyRosetta loss names |
| Independent RNG and missing-dependency guard | Actual absent dependency | Exactly identical | Python, NumPy, Torch CPU state before/after app load, classification, and two guard calls; same missing-dependency message and original cause |

Candidate total: **22 tests passed, zero skipped** (18 recovered + 4 existing).
The baseline's 8 failures and separate collection error are preserved as expected
feature red results, not described as scientific regressions. Full raw logs/JUnit
and the unmodified two original test files' hashes are archived. The independent
snapshot's before-import attempts are `["pyrosetta"]`; the candidate's are `[]`.
Local proprietary PyRosetta was absent, so this table alone is not native validation.

Exact candidate pytest suites:

```text
tests/test_pyrosetta_lazy_import.py
tests/test_runtime_requirements_source.py
tests/test_loss_import_lifecycle.py
tests/test_loss_registry_chain_pair_losses.py
```

The first two were run together (18 passed in 17.24 s); the last two together
(4 passed in 5.22 s), using the existing geometry venv, CUDA hidden and CPU threads
set to one. These are local test durations, not benchmark comparisons.

## Real SIF diagnostic, after official W0 completed

Root verified and transferred two new isolated clones with the exact HEAD/tree
above. A read back the existing CPU-only `control/test-container-cpu` wrapper,
then used it unchanged with `/alphafold3_venv/bin/python -X importtime` and the
adjacent `profile_cold_startup.py`. The wrapper hides CUDA/HIP/ROCR, uses one CPU
thread per numerical library, disables GPU preinitialization, and uses the already
installed licensed SIF. The actual argv, source identity, start/end time, output
hashes, process results, and unmodified stderr traces are in each raw record.

Four fresh processes ran in **before, candidate, candidate, before** order. There
were no warmups, retries, native initialization, trajectories, forwards, or samples.
Each Python process had a 19 s soft / 20 s hard CPU limit; each enclosing command
had a 35 s wall ceiling and a 2 s termination grace. All four exited zero.

| Invocation | Runner load wall / CPU (s) | First source scan wall / CPU (s) | Whole invocation wall / CPU (s) |
|---|---:|---:|---:|
| Before 0 | 10.6619995 / 5.4203785 | 1.1830352 / 1.1741641 | 14.3738128 / 14.0384830 |
| Candidate 1 | 7.7178666 / 3.6088793 | 0.0615744 / 0.0578269 | 9.4969621 / 8.3665780 |
| Candidate 2 | 6.5846493 / 3.5389525 | 0.0617199 / 0.0587626 | 8.1983432 / 8.3447780 |
| Before 3 | 9.6816375 / 5.3647679 | 1.1912973 / 1.1843292 | 13.2845317 / 14.2356180 |

The four whole invocations consumed **44.985457 CPU seconds** and **45.353650 wall
seconds** in total. CPU accounting includes the host controller and reaped wrapper
children, so it is intentionally broader than the two in-Python phases. It remains
below the authorized 90 CPU-second diagnostic ceiling. Scientific evaluations: **0**.

The original importtime traces identify actual avoidable native cost: baseline
`pyrosetta` cumulative time is 3.216977 / 3.008356 s, containing
`pyrosetta.rosetta` self time of 3.054814 / 2.829178 s. The candidate contains no
PyRosetta import and `sys.modules` confirms that fact. Torch remains present in
both sources, at 3.50–3.95 s cumulative import time; it is not counted as a saving.
Nested cumulative import times are not added together. All four runs retain all
169 registry entries and avoid `scipy.spatial` on cold sequence-only startup.

These observations support the two caller hypotheses. They do **not** establish
valid completed-work throughput: there are only two cold observations per source,
no trajectory execution, and no scientific acceptance gate in this diagnostic.

## Independent native science gate and remaining decision

After A's fourth process ended, C alone used the native slot. C reports two actual
PyRosetta processes PASS, totaling 30.020256 host/children CPU seconds, with no CUDA
initialization. Its separate fixed 27-residue three-chain control checks actual
SASA, `chain_buried_fraction`, `com_distance`, all atom coordinates, native first-use
and explicit initialization boundaries, three RNG streams, and full registry and
requirement metadata. Exact equality passed. This is a narrow installed-native
gate, not certification of every PyRosetta loss or a changed W0 workload. C's raw
evidence is owned in `control/f102-startup-science-review` and the corresponding
root delivery; A does not present C's execution as its own.

Root has started the unchanged full W0 comparison at
`control/w0-startup-f9-vs9d-01`: 30 trials, 90 trajectories, 270 evaluations, 900 s
wall and 260 MB output budget. No new native/profile process is authorized while
it runs. KEEP still requires complete work, science and release gates, and an
independently audited improvement beyond noise. Otherwise revert the candidate
and retain its evidence. The preceding broader Runtime correctness changes had
no reproducible throughput gain in the completed same-host W0, which remains a
separate negative performance result.

## Evidence integrity

The complete local review directory is in `raw.tar.xz`, with every archived file's
byte length and SHA-256 in `raw-manifest.json`. It includes local red/green logs,
source provenance, registry snapshots, native command schema/status, exact probes,
the historical prepared profile plan, and all 17 actual remote diagnostic files.
The original native-only archive is retained inside it: 45,256 bytes, SHA-256
`df808b1046394c08e77f01b89e65674e1f252facb42323c51802cecf5530c381`.

The native profile child script SHA-256 is
`9d1580d22c779677527154596d5295b2df89a83acd0da516beb425a8f359be9b`.
The observed existing CPU wrapper, rather than the illustrative raw Apptainer
command in the initial prepared plan, is the actual execution entry. The archived
per-command records are authoritative. Neither profile data nor CPU fake-symbol
tests are labeled GPU acceleration evidence.
