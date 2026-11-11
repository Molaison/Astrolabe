# Mol-1: automatic base/Constraint native adapter

`mol1` and `mol-1` select the same family factory and existing `NativeModelClient`.
The source is pinned to [official Constraint commit 9e85ad16](https://github.com/molence/Mol-1/tree/9e85ad16b3d7031dd92c34d99f895ee65547038b).
The default `model_name: auto` selects `mol-1-base` without active constraints,
and `mol-1-constraint` with active constraints or enabled dynamic binding-site hints.
Each model has its own checkpoint; neither substitutes for the other. The official
base loader uses strict loading and the Constraint loader strictly checks all model tensors.
Before automatic selection was added, the 2026-11-11 Constraint functional canary
passed on an RTX 3090 with the official
deterministic mode: native CLI and this adapter matched all 1204 labeled atom
coordinates and compared numerical confidence fields exactly. This covers two
76-residue ubiquitin chains, SITE/CONTACT/EXCLUSION, seed 101, bf16, 2 cycles,
5 steps, one sample, no MSA, torch triangle kernels, no fusion/TF32, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=101`. It does not qualify
default-budget inference, other inputs, throughput or recovery fault scenarios.
The automatic-selection revision was then verified on the main integration tree:
one real runner completed base → Constraint → base, each result matched its own
native reference for all 1204 labeled atoms and compared confidence fields with
maximum difference zero. Same-selection access reused the client; switching
stopped the previous private process and created a new residency. Thirteen CPU
selection/input routes and fifteen existing registry/owner checks passed.
The non-deterministic comparisons did not pass the same preregistered
tolerances; their failed evidence is retained, not treated as adapter equivalence.

Evidence: [automatic selection and main merge](/xcf26/zpzeng/practice_arena/224_astrolabe_architecture_handoff/publish/001_execution_architecture/01_shared_contracts/57_mol1_auto_model_main_merge/README.md),
with the [earlier Constraint qualification](/xcf26/zpzeng/practice_arena/224_astrolabe_architecture_handoff/publish/001_execution_architecture/01_shared_contracts/55_mol1_constraint_integration/README.md) retained separately.
Merging this backend into main does not retarget the machine's existing daily
launcher or running jobs, nor does it grant inference-service/Beam/Mutate/Score
performance qualification.

## Configuration and invocation

External paths are configurable through `subprocess_envs.mol1.{python_bin,repo_path}`,
`weight_path.mol1` (shared data root), `weight_path.mol1_base`, and
`weight_path.mol1_constraint` (separate checkpoints), or the
corresponding `evaluator.mol1` fields `python_bin`, `repo_path`, `mol_root_dir`,
`base_checkpoint_path`, `constraint_checkpoint_path`. Environment overrides are
`GH_MOL1_PYTHON`, `GH_MOL1_REPO`, `MOL_ROOT_DIR`, `GH_WEIGHT_MOL1_BASE`,
`GH_WEIGHT_MOL1_CONSTRAINT`. Host defaults point to the prepared
`mol1-constraint-v1` runtime and the existing `mol1-evaluator-v1/venv/bin/python`.
Both default checkpoint paths are below the shared root's `checkpoint/`; the
prepared base path is a read-only symlink to the original base checkpoint.
The child uses the configured source first on `PYTHONPATH` and `LAYERNORM_TYPE=torch`;
it does not install packages or change the shared environment/launcher.
Source, data root, checkpoint, reference files and configured extras are expanded
and made absolute in the host before the child's cwd changes to the source tree.
An explicit `python_bin` path is expanded/made absolute **without resolving its
executable symlink**, preserving the venv entry and its site-packages.

The existing Mol-1 venv alone need not contain Astrolabe's Python dependencies.
`evaluator.mol1.extra_python_paths` (defaulting to
`subprocess_envs.mol1.extra_python_paths` / `GH_MOL1_PYTHONPATH`) accepts a list
or an OS-path-separated string. The host default is the new runtime's
`python-extras` directory, supplied separately by the parent (including missing
`omegaconf` and other pure runtime dependencies). Existing parent `PYTHONPATH`
is also inherited. Resolution order is pinned Mol-1 source, this Astrolabe
checkout, configured extras, inherited paths. This is path wiring, not proof
that the full child dependency set imports successfully.

`evaluator.mol1` also exposes `num_cycles`, `num_steps`, `num_samples`, `num_workers`,
`dtype`, `device`, `use_msa`, `triangle_multiplicative`, `triangle_attention`,
`enable_fusion`, `enable_tf32`, `deterministic`, `timeout_seconds`, `constraints`,
and `reference_assets`. Full atom confidence is required (`need_atom_confidence: true`);
`execution_mode` must be `daemon`. Defaults 20 cycles / 200 steps / 5 samples / bf16
follow the release, not a performance recommendation. These parameters are passed
explicitly to `batch.get_default_runner(..., provided_options=None)` (the Python default);
no global `inference_configs` mutation or CLI reload is used.

### Selection and compatibility

```yaml
evaluator:
  mol1:
    model_name: auto
    base_checkpoint_path: ${weight_path.mol1_base}
    constraint_checkpoint_path: ${weight_path.mol1_constraint}
    load_checkpoint_path: null
    dynamic_binding_site: false
```

`auto` and explicit `mol-1-base` both upgrade to Constraint for configured
`constraints`, late `Input.constraint_ir`, or the merged `binding_site`.
`dynamic_binding_site: true` or global `dynamic_binding_site_enabled: true`
selects Constraint even before a dynamic site is available, matching Protenix.
Static binding guidance is an initial factory hint only: each request rechecks the
real merged job. Static/per-oligomer binding still takes precedence over dynamic
fallback. Null, `{}`, null/empty primitives, SITE/EXCLUSION with `residue_ids: []`,
and CONTACT with empty `pairs`/`token_pairs` are inactive as in the official parser.
Unknown fields or malformed primitives fail; inactive primitives are removed before
merging, so they cannot mask or conflict with an active source.

Explicit `model_name: mol-1-constraint` remains Constraint-only, including empty jobs.
Legacy `load_checkpoint_path` overrides the corresponding named model's path; with
`auto`, its filename must identify `mol-1-base.pt` or `mol-1-constraint.pt`.
An explicit base configuration with a legacy base path still upgrades using the
separate `constraint_checkpoint_path`. Checkpoint filenames and strict native tensor
loading guard against swapping model weights. Missing assets fail, never fall back.

Startup and `local_assets(settings)` (also used by the durable selected-asset manifest)
require **both checkpoints** for auto/base, since late input can select either.
Constraint-only requires just its checkpoint. This is config-wide asset availability,
not model residency: unconditioned auto jobs load only base. The runner freezes
the full config-wide content identities on first preparation and rechecks them;
each actual `NativeModelClient` binds only the selected checkpoint plus shared
source/adapter/reference assets. `local_assets(settings, model_name=...)` enumerates
that per-model set. No late input can introduce an unfrozen checkpoint.

The same `Mol1Runner` retains a loaded client for an unchanged selection. On
base → Constraint → base it first calls normal shutdown, then constructs the next
client with the selected model/checkpoint/asset identity. It does not keep a
two-model cache. The existing `NativeModelClient` adds an opt-out `poolable=False`
for Mol-1: it bypasses both local cross-trajectory pooling and automatic attachment
through `ASTROLABE_RESIDENT_MODEL_REGISTRY`. Without an explicit `socket_path`
(Mol-1 supplies none), it always owns a private daemon, even with
`ASTROLABE_MODEL_OWNER=trajectory`; normal shutdown stops that owned daemon before
the replacement loads. Existing shared servers are not stopped or modified.
Explicit socket attachment remains supported for callers that request it;
`poolable=True` retains the existing pooling/registry/trajectory behavior unchanged.
Failed native requests remain
fenced, including unknown outcomes; switching or ordinary shutdown cannot clear
that fence. Failed cleanup retains the old client and blocks replacement.

### Parent acceptance/observation API

`mol1_contract.selected_model(settings, job=None, *, constraint_hint=False)` returns
`mol-1-base` or `mol-1-constraint`. Supply the **merged official job**, not a folding
Input, for request selection. `checkpoint_paths(settings)` returns the required
model-to-`Path` mapping. `Mol1Runner.from_config(conf, *, constraint_hint=False)` and
`run_inference(input_json=..., out_dir=...)` retain the existing construction/execution
entrypoints. `prepare_shared_model(job=None)` returns the selected native client;
without a job it selects initial configured residency, so it is not a passive observer.

After each real request, inspect without loading or switching:

```python
client = runner.native_client  # None before first preparation / after successful shutdown
model = client.metadata["model_name"]
checkpoint = client.metadata["checkpoint_path"]
identity = client.owner.identity  # model/checkpoint/code/reference content-bound identity
residency = client.metadata["runtime_lifecycle"]["model_id"]
pid = client.pid
```

The daemon reply also carries and the adapter checks `model_name`. A new residency
gets a new client/model_id; returning to unchanged base assets may correctly restore
the same content identity as the first base request, not the Constraint identity.

For example, save a standalone task as `mol1_task.yaml`:

```yaml
task_name: mol1_constraint_example
out_dir: /tmp/mol1_constraint_example
seed: 101
steps: 1
mutation_num: 0
enable_pyrosetta: false
skip_complex_poses: true
types: {A: p, B: p}
seqs: {A: ACDEFGHIKLMNPQRSTVWY, B: YWVTSRQPNMLKIHGFEDCA}
evaluator:
  evaluator_terms:
    Mol1:
      _type: mol-1
      num_samples: 1
      deterministic: true
      constraints:
        site:
          source_chains: [A]
          partner_chains: [B]
          residue_ids: [[A, 3]]
        contact:
          pairs: [[[A, 3], [B, 8]]]
        exclusion:
          source_chains: [A]
          partner_chains: [B]
          residue_ids: [[A, 15]]
mutator:
  mutator_terms:
    MCMC: {_type: MCMC}
loss:
  loss_terms: {}
stages:
  default:
    evaluator: Mol1
    oligo: [AB]
    mutators: [MCMC]
    mutation_num: 0
    weighted_score_expression: "0"
    acceptance_threshold: null
    early_stopping: null
```

Run from this checkout using the existing Astrolabe interpreter:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=101
python -m astrolabe.cli run --standalone /absolute/path/mol1_task.yaml --dry-run
python -m astrolabe.cli run --standalone /absolute/path/mol1_task.yaml
```

These are invocation examples, not full-pipeline acceptance commands. The
pre-auto documented dry-run passed; real model qualification used the frozen canary
parameters above, not this example's release-default cycles/steps.
The backend catalog and interactive evaluator selector expose `Mol1`.

## Input and confidence contract

| Input / output | Adapter capability (not native qualification) |
|---|---|
| Protein, explicit seeds | Canonical amino acids plus X; physical chain IDs must match the Astrolabe oligomer. Seeds are unique explicit uint32 values; no replacement/default seed. |
| Supplied MSA | `ProteinChain.unpaired_msa` / `paired_msa` A3M text is written to separate native files per chain, with `use_msa: true`. Every chain needs an MSA; there is no automatic search. MSA supplied with `use_msa: false` fails. |
| SITE / EXCLUSION | Exactly `source_chains`, `partner_chains`, `residue_ids`; each residue is `[physical_chain_id, positive 1-based label_seq_id]`. Source/partner chains are disjoint. |
| CONTACT | Exactly `pairs: [[[chain, label_seq_id], [chain, label_seq_id]], ...]` or `token_pairs: [[zero_based_native_token, zero_based_native_token], ...]`, never both. Native feature projection remains authoritative. |
| Constraint transport | `evaluator.mol1.constraints`, alias-specific `constraints`, or optional `Input.constraint_ir` using the same official inline mapping. Distinct primitives merge; duplicate primitives across sources fail rather than override. |
| `binding_site` | Existing static/per-oligomer parsing and chain/protomer selectors; static config takes precedence over dynamic fallback as in Protenix. Pocket maps to SITE; contact maps the existing cross-product to CONTACT. `max_distance: 4.5`, `min_distance: 0` are required; arbitrary bounds (including the generic 6 Å default) are rejected. Contact/both needs positions on both chains. |
| Unsupported | Nonprotein, PTM, templates, user CCD, covalent/cyclic topology and unknown configuration/constraint fields fail explicitly; active constraints never fall back to base. |
| pLDDT | Whole-complex native 0–100 is divided by 100; native atom pLDDT is already 0–1. Real per-residue values average atom confidence through native token/residue identities, never repeated chain means. |
| PDE / PAE | `StandardScores.pde` receives native contact-weighted `gpde` (distance error); `pae=None`. Native pTM/iPTM and chain-pair matrices retain their own semantics. |
| Sample order | Exact requested seed/sample sets are required. Canonical structure is first explicit seed, sample 0, not best-of-N. Native confidence-ranking filename remapping is disabled. Raw files remain under the alias-specific request directory. |

Each seed has a `native_identity.json` built from **actual native atoms and full
confidence mappings** before dumping. Every sample must agree with that axis;
CIF atom IDs and sequences must match it and the input, including repeated-sequence
chains. Chain-pair matrices are mapped through actual asym IDs back to input chain
order. Missing, unexpected or duplicate outputs fail inside the native kernel before
it publishes success; a native `_wounresol.cif`, if present, is a derived structure,
not an additional prediction. Existing owner failure, unknown and cleanup protocols
apply; there is no adapter retry, alternate broker, lock domain or recovery store.

Source HEAD, actual source/adapter Python content, checkpoint and the four consumed
reference files are hashed into native identity and the existing durable selected-asset
manifest. **Replacing reference assets is unsupported.** `reference_assets` may
be empty, or explicitly name the same resolved default files for
`ccd_components_file`, `ccd_components_rdkit_mol_file`, `pdb_cluster_file`,
`obsolete_release_data_csv`. All four reject any other resolved path, including
a different file with identical contents. Defaults are the official `common/`
files below `mol_root_dir`; symlinks to those same files are accepted. This
restriction prevents config overrides from diverging from the default CCD/RDKit
paths cached by the official module at import. Assets must already exist locally.
The native operation retains the official data/model/terminal-geometry/
dump/cleanup order as one composite admitted operation; its wall time is not GPU
kernel time. Concurrency is not certified and scheduling policies are unchanged.

Conditioning does **not** guarantee satisfaction. The parent must separately qualify
native equivalence, failure/recovery and scientific chain/sample identity, then measure
the inference service with prepared real inputs and a continuously ready queue before
testing Beam/Mutate/CPU Score supply. No prior base canary transfers that qualification.
