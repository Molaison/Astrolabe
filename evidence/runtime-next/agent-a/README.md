# Agent A: runtime, measurement and R1 evidence

This directory packages the authored and independently rerun evidence without
changing production source. `agent-a-raw.tar.xz` preserves the full local raw
tree losslessly. `raw-manifest.json` records every archived relative path, byte
count and SHA-256, plus the archive hash. Reports, key probe scripts and the
failed R1 patch remain readable alongside it. Local originals remain in the
author's worktree; they were not overwritten or deleted when packaging.

Extract into a new directory to inspect all stdout/JUnit/JSON/config/trace files:

```sh
mkdir agent-a-evidence-review
tar -xJf agent-a-raw.tar.xz -C agent-a-evidence-review
```

The archive uses a `raw/` prefix. Verify its bytes against the manifest before
relying on derived results. Reproducing old source requires the exact commit
specified by each report/probe; failures measured before commit also record the
functional source hash or patch. A source HEAD alone does not certify uncommitted
code, and those measurements explicitly retain the source state.

| Evidence group | Reviewable entry point | Scope/status |
|---|---|---|
| Incremental architecture investigation | `INVESTIGATION.md` | Historical traces are labeled; old CPU regression is not a new measurement |
| Deadline and delegation correctness | `RUNTIME_CORRECTNESS.md`, `probe_deadline.py` | Actual State/ledger locks, controlled worker boundary; no model/IPC/GPU claim |
| Frozen measurement and CI entry | Raw `ci-plan/`, `ci-local-preflight/`, harness logs, `cpu-torch-job-local.json` | Plan-only files contain no trials; local IPC/PID limits are explicit |
| Independent C review | `review-c/REPORT.md` | 57 passing scope tests, ten reproduced invalid-input failures, retained blocking-backpressure negative |
| R1 spatial dependency candidate | `R1_SPATIAL_IMPORT.md`, `check_r1_source.py` | Candidate correctness passed; performance awaits frozen CI |
| Failed two-module R1 attempt | `r1-failed-two-module-candidate.patch`, raw `r1-residual-import.log` | Exact failed source reconstructed and verified against recorded source SHA; not retained as production |

R1's **67** passing cases cover seven suites: root's independent five-suite
**39**, plus sequence losses **25** and registry chain-pair losses **3**.
`r1-test-scope.json` contains the exact JUnit-derived accounting. They are
different executions and must not be added as distinct test coverage.

Two additional direct W0 import/correctness probes ran **18 evaluations total**,
three repeats × three steps each, before and after the unsuccessful two-module
attempt. They preserved exact scientific rows and all 141 then-available
registry entries. They were not timing trials, hidden warmup, a resident run, or
part of the fixed 30-trial performance comparison. In the isolated geometry
environment, all 169 registered names/metadata match and real PDB/SciPy controls
pass. The archived install reports/freezes make that environment explicit.

At packaging, R1 production commit is `605bad02f5c71083a95142f03bebc0db24a27d84`,
CI wiring is `9ddd39ea38c77587b7e87c4cb9357bdd9c4e51e6`, and the parent is running
the published candidate `71c7592` in CI `41764768462`. No unfinished CI result,
pretrained backend certification, GPU speedup or 20% performance target is
claimed by this evidence snapshot.
