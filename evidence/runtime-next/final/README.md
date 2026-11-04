# Final verification at 9d62b3e

**CPU verification DONE; overall task PARTIAL.** The exact source/PR HEAD is `9d62b3eb86cb75a2266af49dc48cb851e69e0341` (tree `bb2cc1ce9b9ad2b17585f581bca86d8dd7d729fc`). These completed validation records are attached on the separate `evidence/runtime-final-validation-20261104` branch. Product, config, test, benchmark and CI sources remain identical to the tested PR HEAD.

| Gate | Actual result | Raw/evidence |
|---|---|---|
| [Runtime CI 41766956074](https://github.com/Molaison/Astrolabe/actions/runs/41766956074) | 248 runtime tests, 104 CPU Torch/structural tests, no skips | `../ci/cpu-ci-04-summary.json`, `../ci/torch-ci-03-summary.json` and adjacent lossless archives |
| Frozen W0 end to end | 30 valid trials, 90 completed trajectories, 270 evaluations; raw CSV exact, releases confirmed | `../ci/runtime-cpu-9d62b3e.tar.xz`; original artifact 13293088703, ZIP SHA-256 `d46330158d79a18da6df6813b0d9a5f8779c325547fa1951f98af541640cccc8` |
| CPU Torch/structure environment | Actual CPU Torch 2.7.1+cpu, CUDA build null; real process tests use synthetic weights | `../ci/runtime-torch-9d62b3e.tar.xz`; artifact 13293197994, ZIP SHA-256 `5618e1d3c6395075905762636f919950a06fabb935ac6c82e7f868d5440b74c5` |
| Lead's raw W0 audit | 90 accepted initialization rows, 180 rejected mutation attempts, 0 accepted mutations; all 15 same-mode pairs improve with separated ranges | `w0-root-audit.json`; reproducible `../review/audit_w0_root.py` |
| C's independent complete raw audit | DONE/errors=[]; all source/config/trace/CSV/provenance/telemetry verified, same conclusions | `independent-review/report.json`, input manifest and recorded invocation; reuse existing reader without modifying source |
| Actual CLI scientific positive control | 4 evaluations, 5 rows, 1 accepted mutation, 1 stage entry per revision; exact rows/owner/CPU RNG/output hash | `scientific-control-summary.json`, `scientific-control-9d62b3e.tar.xz`, per-file manifest |
| Backend matrix | 4 tests passed; full registry regenerated at final source HEAD | `backend-matrix-tests.xml`, `backend-certification.json` |
| Python 3.11/3.12 distribution | Push and PR jobs all succeeded | `github-runs.json`, runs 41766956068 and 41766961493 |

Final same-runner aggregate valid evaluation throughput improves **11.55% direct, 13.27% resident serial, 13.42% resident adaptive**. Median wall reductions are 8.94%, 11.08% and 12.41%. The paired-median statistic is separately recorded by C; it is not substituted for the aggregate metric. Resident still takes about 2.5 times direct, and the original 20% objective is not met. This is a second fixed-protocol CPU comparison; absolute times from different CI runs are never divided as speedup.

The separate positive control does not enter performance timing. No result certifies pretrained GPU inference, coordinates or checkpoint/resume. GPU/weights/allocation and the requested `/xcf26/.../cloud-20261104` destination remain BLOCKED. The integrated candidate contains earlier correctness changes as well as lazy imports; no per-commit causal ablation was performed.

The parent [evidence index](../README.md), [final report](../../../docs/runtime_final_report.md), `verification.json` and `publication-map.json` connect source, independent review and archives. From `evidence/runtime-next`, `sha256sum -c archives.sha256` verifies all raw archives before extraction.
