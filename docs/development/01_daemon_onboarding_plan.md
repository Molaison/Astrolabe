# Installation and resident model execution

This change makes installation and a first run discoverable from `astrolabe`.
Model execution must reuse loaded weights across requests. A process that starts
a fresh model CLI for every request does not satisfy that requirement.

## Acceptance

- A new user can inspect an installation plan, install a selected task profile,
  diagnose missing assets, and run a bundled CPU example from the public CLI.
- Installation reuses the existing software catalog, asset validation, and
  container builder. Restricted model access remains an explicit user action.
- Every registered model backend has a recorded execution policy. Existing
  resident runners remain resident. Subprocess backends gain a persistent model
  worker rather than a wrapper around repeated subprocess inference.
- Worker startup, multiple requests, request failure, timeout, shutdown, and
  model identity are observable and tested. Accepted inference is not replayed
  automatically after an ambiguous transport failure.
- Anthropic integration pins its source and checks the requested mode. A
  different model or unsupported optimization fails visibly instead of silently
  changing a scientific protocol.
- CPU behavior and packaging checks pass. GPU correctness or throughput is
  reported only for a measured workload on an allocated GPU.

## Scope and evidence

The starting checkout contains ongoing interface-refinement changes. They remain
part of the working tree. The initial tracked diff and dirty-file hashes are
saved locally at `/tmp/astrolabe-initial.patch` and
`/tmp/astrolabe-initial-dirty.json` for review of this change.

The existing backend registries are `astrolabe/evaluators/utils.py`,
`astrolabe/mutators/base.py`, `astrolabe/structure_processors/base.py`, and
`astrolabe/initializers`. Existing AF3, RF3, DPLM, SurfDesign, SurfPro, and RAPiDock
services provide reusable lifecycle patterns. ProteinMPNN and several predictors
already retain model objects in their owning process.

The initial focused CPU check reported 145 passed and four failures. All four
failures require the unavailable project fixture
`inputs/001_binder_interface/01_astrolabe_beam/03_bind_target_homologies`.
This is the pre-change baseline, not evidence for a new regression.

## Design inputs

Official sources were checked on 2026-10-31.

| Source | Adopted lesson |
| --- | --- |
| [Boltz](https://github.com/jwohlwend/boltz) | Show the package install and a concrete prediction command together. Keep asset caching explicit. |
| [Chai](https://github.com/chaidiscovery/chai-lab) | Provide an immediately runnable example and name model download behavior. |
| [Ollama lifecycle](https://docs.ollama.com/faq) | Expose loaded-model state and explicit release. Bound memory use rather than keeping every model loaded. |
| [Anthropic kits](https://github.com/anthropics/uplifting-biomolecular-modeling) | Pin each upstream stack, activate a named mode, and refuse failed activation. |

`config/anthropic_kits.yaml` pins revision
`f4f62fa6592ae4938d49b1757bea0cfeff9f468e`. The upstream repository describes
itself as an unmaintained reference release. Upstream claims about `exact`,
`fast`, and `big` are not Astrolabe validation. The default for a new optimized
path must preserve the configured model and expose numeric tradeoffs.
Boltz 2 is not an implicit replacement for the current Boltz 1 evaluator.
ESMFold 2 is not an implicit replacement for ESMFold.

## Execution order

1. Record the existing registries, lifecycle behavior, and installation paths.
2. Compare two concrete architecture sketches and select a shared contract.
3. Implement the public installation, diagnostic, and example commands.
4. Implement persistent worker transport and migrate subprocess model owners.
5. Connect compatible Anthropic optimization modes and identity checks.
6. Review the actual diff, run public CLI and worker lifecycle checks, and run
   the relevant CPU regressions. Record remaining hardware validation gaps.

## Selected architecture

The two sketches converge on a runner boundary. The independent review chose
the second sketch's explicit backend adapters. The shared owner receives only a
backend request. Existing feature preparation, scoring, candidate selection,
and trajectory state stay in their current modules.

The first implementation uses `DaemonSpec` to name one backend factory, its
immutable settings, interpreter, and environment. `OwnedModelDaemon` loads that
factory once, handles repeated requests, exposes status, and drains on shutdown.
Backend factories keep their model or engine in a closure or runtime object.
An explicit attachment validates identity and does not transfer stop authority.
This is separate from scientific model replacement and optimizer qualification.

The registry includes neural structure processors as well as predictors and
mutators. RAPiDock's present JSONL wrapper still launches an inference CLI inside
its daemon. It needs real score and confidence model ownership. Native resident
objects already avoid per-step weight loads, but do not establish external
daemon or cross-trajectory reuse by themselves.

The upstream `af3_jax` kit uses a converted OpenFold3 weight variant. Its current
manifest mapping does not establish compatibility with licensed AF3 weights or
Astrolabe's cyclic patches. Selection must preserve that distinction.

## Throughput checkpoint

- Blocking first steps are the backend inventory, model compatibility rules,
  and worker protocol. They precede adapter changes.
- Independent workstreams own installation, worker transport, and separate
  backend adapter files. The coordinator owns CLI registration and integration.
- Shared mutable state is limited to each worker's model state. Requests to one
  model serialize. Separate workers have separate output paths and write sets.
- The smallest safe decomposition reuses existing resident implementations and
  adds persistence where loading currently repeats. It does not add an unrelated
  distributed scheduler.

Task state is tracked in `Astrolabe-izo`. Decisions and validation pointers are
recorded in `02_daemon_onboarding_decisions.tsv` in this directory.

## Shared scientific contracts

The architecture discussion also identifies a domain layer above model
execution. This section is a design proposal. The daemon migration continues
to use the existing scientific result, sequence, and mutation APIs.

`StandardScores` currently combines structures and model confidence fields.
`ScoreContribution` lets additional components emit named values. Both are
useful migration inputs. A common metric contract should identify the producer,
meaning, unit, subject, shape, index mapping, candidate state, prediction sample,
and dependencies. Prediction, assessment, and mutation results retain their own
types and carry a shared metric collection.

Raw measurements and objectives have different responsibilities. Shared metric
functions calculate geometry, energy, confidence, and sequence properties.
Loss terms define transformations, constraints, aggregation, and weights.
Selectors, mutators, and reporting can consume the same measurements without
depending on the loss implementation. Objective direction belongs to the
configured scientific task.

Shared segments remain canonical sequence variables. Protomers are ordered
compositions of segment references. Chains are instances in a particular
assembly or evaluation context. Model chain and residue indices map explicitly
to canonical segment sites. Multiple references to one segment retain shared
sequence semantics.

Mutation proposals identify their base state, canonical sites, expected original
tokens, proposed tokens, producer, and supporting metrics. The design loop
creates a candidate state and owns acceptance. Stacked mutators can read an
explicit staged candidate. Conflicting proposals require a declared combination
or selection policy. They cannot silently overwrite the same shared site.

Metrics record their actual dependencies. Sequence-dependent measurements,
structure-dependent measurements, and sample-specific model confidence need
different invalidation rules. Reuse requires matching dependencies. Missing or
stale measurements are explicit states and cannot become numeric zero.

This shared domain layer is a library and a per-run state owner. Expensive model
or tool implementations can use isolated workers underneath it. Shared score
semantics do not require a separate score daemon.
