# Real CLI scientific positive control

**Status: DONE for this separate CPU correctness control.** Baseline `d66ee714f2e755948534f95d9d125287ce8dbd91` and integrated candidate `db937f9706270cb109e799cda29f52d2872726b5` produce exactly the same proposals, scores, acceptance decisions, stage transition, terminal scientific owner, Python/NumPy/torch CPU RNG states and `scores.csv` bytes. This is not a throughput benchmark, GPU/pretrained certification, checkpoint-resume test or extension of frozen W0.

## Why this control exists

The formal frozen W0 comparison had no accepted mutation attempts. Initial accepted rows do not demonstrate optimization progress. Existing actual CLI runtime fixtures also start at the optimum of the alanine-fraction objective. Existing stage unit tests do not fill this real-CLI coverage gap. This new, explicitly separate four-step control exercises acceptance and a subsequent stage transition through the public CLI without patching the scientific implementation.

The original production MCMC mutator, sequence-residue-fraction loss, Dummy evaluator, decisions and stage transaction run unchanged. The independent fixture maximizes negative alanine fraction, starting from `AAAA`; one accepted mutation makes `alanine_fraction < 1.0` and triggers the second stage. `write_structure: false` avoids requiring PyRosetta. No score, mutation or acceptance function is mocked. The result remains a CPU/Dummy control, not real model inference.

## Frozen protocol and budget

New source files: `benchmarks/runtime/workloads/cpu_scientific_control.yaml` and `scripts/validation/runtime_scientific_control.py`. No production files or shared W0 measurement files were changed.

| Frozen item | Value |
|---|---|
| Workload SHA-256, both rounds | `ddbd295188ed22241736750bed4fa613c553dbf29b5423de0f9454b7c90cbde9` |
| Final harness SHA-256 | `6f52867a3d45e83985fc84b844cfdf7227e61761e615355a63b913d83eb4ed26` |
| Final plan SHA-256 | `fdcd2f47e7f51b724ac6f661bf8fc6f10ce4905c7a706799910575abf89f4bae` |
| Each source, each round | One repeat; seed 17; steps 0–3; three mutation attempts |
| Actual backend calls per run | Four: initial plus three mutation evaluations |
| Scientific score rows per run | Five, including one stage score recomputation |
| Limits per run | 45 seconds; 16 MiB outputs; direct mode; GPU hidden; CPU library threads 1 |
| Timing policy | No throughput, end-to-end elapsed-time or speedup metric collected |

The harness freezes production source HEAD, configuration-tree hash, workload, itself and the four existing completed-trajectory reader dependency hashes before execution. The production/configuration worktrees must be clean and remain unchanged. A detached candidate snapshot isolates the comparison from later integration commits. Each child invokes the actual `astrolabe` module CLI, then reads surviving scientific owners and already-loaded CPU RNG libraries. This observer is checked not to consume RNG.

## Actual result

Both sources produce the following identical trajectory:

| Step | Proposed sequence | Weighted score | Mutation decision | Stage outcome |
|---:|---|---:|---|---|
| 0 | `AAAA` | -1.00 | Initial state | `default` |
| 1 | `GAAA` | -0.75 | Accepted | Enters `accepted_mutation`; recomputes score -0.75 |
| 2 | `SAAA` | -0.75 | Rejected | Remains `accepted_mutation` |
| 3 | `AAAA` | -1.00 | Rejected | Remains `accepted_mutation` |

The terminal accepted sequence is `GAAA`, `last_accepted = 1`, accepted steps are `[0, 1]`, all four proposals remain in the decision owner, and no transition gate is active. All score components, decision diagnostics and stage/acceptance rows compare exactly, not merely final score or sequence.

Terminal Python, NumPy and torch CPU RNG states are saved in full and compare exactly. Canonical JSON hashes, for convenient cross-checking, are Python `a52fa02f3c436257d6a443113d94fb61af3f71b347fdfd4900e854b9a959ac3e`, NumPy `63a9c4ee56e0af810b4df9a46c39114983644d8afc4473c9b852f48f270d838e`, and torch CPU `4f8aab5aa5b2904ec9a6c31da6e673a25a52f42345831ffc4cc5a0ca3b6bcabb`. These hashes use sorted-key JSON with compact separators. The `scores.csv` SHA-256 is `ae24138cfada92ded7c0251cd160b42f18b1f8f1070f2725aad36e4744812936` on both sources. No structures were generated.

## Preserved negative result and correction

Round `scientific-positive-01` froze the mistaken expectation of five backend calls: initial, three mutations, and one presumed target-stage reevaluation. Both actual CLI runs succeeded and all scientific comparisons already matched, but the checker correctly returned **PARTIAL** because each source emitted only four calls for each of `cpu_prepare`, `cpu_execute` and `cpu_finalize`.

Inspection of the existing `stage_transaction.plan_stage_transaction` and `begin_direct_stage_entry` established the reason: when evaluator alias and oligo are unchanged, the stage transition clones the previous result rather than invoking the backend. `Decisions.recompute_last_score` still emits the row labelled `re-evaluate due to stage change`. A row with that label is not proof of another backend execution.

The correction changes only the checker: four backend calls, five score rows, one stage score recomputation. The workload hash, step/seed/repeat budget, objective and all production code remain unchanged. Round `scientific-positive-02` was frozen in a new directory before execution and returns **DONE**. Scientific rows, terminal owners/RNG and output hashes also compare exactly between the first and second rounds for each source. The lower count is existing baseline result reuse, **not an optimization achieved in this task**. The complete first report, wrong frozen expectation, logs, outputs and `harness-frozen.py` are preserved.

Across the two rounds, four real CLI invocations ran in total: two baseline and two candidate. This is correctness evidence only; neither round is included in W0 or performance samples.

## Reproduction

From a checkout containing the two new files, with committed clean baseline/candidate production sources and the existing CPU dependencies:

```sh
../venv/bin/python scripts/validation/runtime_scientific_control.py freeze --baseline ../baseline --candidate ../scientific-control-current-db937f9 --output evidence/runtime-next/agent-c/scientific-positive-02
../venv/bin/python scripts/validation/runtime_scientific_control.py run --plan evidence/runtime-next/agent-c/scientific-positive-02/plan.json
```

Use a new output directory for a new replay; existing evidence is never overwritten. A final-HEAD independent replay must freeze that HEAD separately while retaining the exact workload hash and scientific budget. The old plan intentionally refuses a different source or harness.

Raw artifacts under each round include `plan.json`, copied workload, `report.json`, and per-source `command.json`, `cli.log`, `runtime.jsonl`, `validation.json`, full terminal scientific/RNG snapshot, provenance and actual scientific outputs. Final `report.json` records all three exact comparisons as true.

## Boundaries

This control validates real CLI accepted-mutation and direct stage-entry behavior for one fixed CPU trajectory, including two rejected proposals and owner restoration. It does not certify MH acceptance, arbitrary stages, rollback/failure recovery, checkpoint resume, asynchronous candidate scheduling, subprocess/broker execution, structures, model weights or GPU numerical behavior. Those retain their own independent tests and explicit limitations. W0 remains unchanged, and this control provides no performance claim.
