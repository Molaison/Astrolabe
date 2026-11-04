# Independent full-raw W0 audit: recovered startup optimization

**Audit DONE.** Exact candidate `f9ce87b9f5cf37710f0a52ebe503595e5d7fdd7f`
and baseline `9d62b3eb86cb75a2266af49dc48cb851e69e0341` completed the unchanged
30-trial CPU W0 protocol on the same native host/container. All 15 same-mode
pairs improved, with disjoint before/after ranges in every mode. Both paired
median and aggregate completed-trajectory throughput gains exceed 20%.
Candidate resident modes still take approximately 3.22 times the candidate
direct time; this result does not establish a resident advantage.

This is Agent C's independent reanalysis of persisted individual observations,
all original CSVs, provenance, full configurations, runtime traces and source
hashes. No new benchmark was run by this review. The reader does not consume
the benchmark author's aggregate performance or scientific conclusions.

## Preserved original evidence

The remote source directory was
`/xcf26/zpzeng/practice_arena/astrolabe_runtime_20261102_implementation/cloud-20261104/control/w0-startup-f9-vs9d-01`.
It contained 634 files and 7,520,683 bytes. All files were archived read-only,
then rehashed; all 634 downloaded files matched their original SHA-256 and byte
length. `w0-f9-raw.tar.xz` contains the original directory and its `MANIFEST.json`.
The archive is 242,004 bytes, SHA-256:

```text
f030e4888f5af99ba46ca43391258a2e92cc67f28d51089ef0d2f600cd5e09b6
```

`RAW_MANIFEST.json` and `audit/input-manifest.json` are byte-identical independent
inventories with SHA-256
`a1b3db521fa2abd5a634a46fce4306f74f4da407039b01f826bf345c073ecf2e`.
The archive retains all successful and diagnostic logs, per-trial telemetry,
runtime events, actual CSVs, configurations, provenance, plan and author report.
`archive-observations.json` preserves native archive execution and completion;
`download-verification.json` records local verification. The parent-owned
freeze-and-invocation artifact was neither modified nor replaced by this review.

## Independent result

The reader found 30 complete trials, 90 complete trajectories and 270 actual
evaluator calls, with zero invalid or failed trial. All 30 full trial configs and
90 executed configs were checked. Each repeat/seed's entire CSV, including every
column and row, matches exactly across modes, sources and groups. The 436
baseline and 437 candidate source-file hashes match the exact Git objects.

All four measurement modules retain their frozen d66 bytes. The complete
scientific configuration, protocol, seeds, work counts and budgets match the
prior CI04 plan. The prior c90-versus-9d plan independently matches ten frozen
fields, including five groups, 30 trials, three trajectories and nine evaluator
calls per trial, a 900-second wall budget and a 260 MB output cap. The protocol
SHA-256 is
`f2636de19cdb90c2707a24969da016c398ba13b71c70fc4cde7e2fa00e7855e5`.

The sum of individual observed trial times is 654.635664894 seconds. All 20
terminal resident broker ledgers are empty. Each observed process tree is
released, with no recorded remaining identity or observation error. Trial times
reconstructed from raw telemetry and process completion agree within 2.4921
microseconds. These checks certify the observer's recorded scope; they do not
prove absence of an arbitrary escaped process outside that scope.

| Mode | Baseline median and range (s) | Candidate median and range (s) | Median paired throughput gain | Aggregate throughput gain | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct | 10.842 [9.292, 14.626] | 6.216 [5.946, 8.097] | +60.2124% | +69.0767% | 5/5 |
| resident_serial | 34.392 [33.014, 38.854] | 19.959 [19.580, 23.720] | +65.4128% | +69.9915% | 5/5 |
| resident_adaptive | 33.629 [31.693, 36.317] | 20.030 [19.371, 25.834] | +63.6063% | +54.9206% | 5/5 |

For each group the paired throughput ratio is baseline elapsed time divided by
candidate elapsed time at equal completed work. Aggregate throughput is total
completed trajectories divided by the sum of observed times in a source/mode.
These are distinct statistics, both calculated from individual trials. The
slowest candidate trial is faster than the fastest baseline trial within each
mode. Five interleaved cold trials per mode on one host support the stated
repeatability; this is not a population confidence interval or a long-run
steady-state break-even experiment.

Candidate resident serial and adaptive take 3.2198 and 3.2340 times the best
direct time on the median paired comparison; neither wins any of its five
pairs against candidate direct. The 20% same-mode CPU engineering objective is
met for this startup candidate. A 20% improvement of resident over the best
direct mode is not met.

## Scientific scope and retained negative results

All 90 accepted rows are initial evaluations. The 180 actual mutation attempts
are all rejected; there are zero accepted mutations and zero stage-entry rows.
For seeds 17, 18 and 19 the same initial `AAAA`, proposed sequences, scores,
decisions, stage labels and complete CSV bytes recur across all comparisons.
Completion and evaluator throughput cannot be read as accepted-mutation or
scientific progress.

W0 uses the CPU Dummy evaluator and contains no full RNG-state snapshots,
structures, pretrained inference or GPU execution. Its parity does not replace
the separate accepted-mutation/stage positive control, the real native
PyRosetta first-use/geometry/RNG control, or checkpoint/retry recovery tests.
The independently verified PyRosetta control for 47c6 versus f9 is preserved in
`../f102-startup/`; it validates two registered native losses and their required
symbols, not every PyRosetta metric. Final integrated-source verification is a
separate root gate: the audited candidate here is exactly f9.

The earlier c90-versus-9d experiment remains preserved separately in
`../f102-startup/w0-c90-raw.tar.xz`. It found no stable additional performance
gain from the broader Runtime correctness fixes: aggregate serial was
−0.1072%, adaptive +0.8966%, direct +4.2869%, with overlapping ranges. That
negative result is not replaced, pooled with, or relabelled as the startup
result. Cross-round or cross-host absolute timing differences are not speedup
measurements. The current evidence supports keeping the startup change while
retaining the resident-versus-direct limitation.

## Reproduce the audit without running a workload

Use a clone containing the exact 9d and f9 Git objects, with the delivered source
bundle if those objects are not fetched by the review branch. The project Python
environment needs OmegaConf. From this evidence directory, first verify and
extract into a fresh directory:

```bash
printf '%s  %s\n' f030e4888f5af99ba46ca43391258a2e92cc67f28d51089ef0d2f600cd5e09b6 w0-f9-raw.tar.xz | sha256sum -c -
mkdir reproduced-raw
tar -xJf w0-f9-raw.tar.xz -C reproduced-raw
```

Verify every original file and the separate prior-c90 workload comparison:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path('reproduced-raw/w0-startup-f9-vs9d-01')
manifest = json.loads(Path('RAW_MANIFEST.json').read_text())
assert len(manifest) == 634
assert set(manifest) == {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
for name, expected in manifest.items():
    data = (root / name).read_bytes()
    assert len(data) == expected['bytes'], name
    assert hashlib.sha256(data).hexdigest() == expected['sha256'], name
before = json.loads(Path('prior-c90-plan.json').read_text())
after = json.loads((root / 'plan.json').read_text())
fields = json.loads(Path('audit/prior-c90-comparison.json').read_text())['all_frozen_fields_equal']
assert all(before[key] == after[key] for key in fields)
print('PASS: 634 original hashes and ten prior-c90 frozen fields')
PY
```

Run the unchanged independent reader through the explicit adapter:

```bash
python review_startup_w0.py --reader ../review_r1_ci.py --repo "$REPO" --artifacts reproduced-raw/w0-startup-f9-vs9d-01 --prior ../f102-startup/w0-prior-ci04 --out reproduced-audit
```

`$REPO` points to that clone. The adapter refuses any original reader other than
SHA-256 `eda968172261e7b266278a4616c65d17b8b70582f3e670b41b7249dff45f5a3a`.
It changes only the expected HEADs and native-host directory layout in memory;
all scientific, timing, source and release checks are unchanged. It writes the
full audit and records the exact adaptations and independent aggregate formula.
Original absolute artifact paths in the saved report are provenance; fresh
reproduction naturally records its own extraction paths.

The raw evidence and report are reusable after an evidence-only cherry-pick.
They must not be silently treated as a measurement of a later production HEAD.
