# RNA secondary-to-tertiary design

Astrolabe can optimize RNA sequence scores before starting a structure predictor. The same stage mechanism also supports a joint objective that retains the secondary score.

## Scientific scope

`rna_ensemble_defect` measures normalized complex ensemble defect. It is the expected fraction of nucleotides whose pairing state differs from a specified target. Lower values are better. An unpaired target nucleotide contributes its unpaired-state probability.

NUPACK's `defect` already returns a normalized value. Astrolabe does not divide it by sequence length again. This score is conditional on the specified complex. It is not test-tube yield and does not include off-target complexes or strand concentrations. [NUPACK definitions](https://docs.nupack.org/definitions/)

Targets preserve strand order and multiplicity. `strands: [A, A]` evaluates two physical copies of the current sequence of protomer A.

For targets with weights \(w_j\), Astrolabe returns \(\sum_j w_j s_j / \sum_j w_j\). Weights must be nonnegative and their total must be positive. The input uses dot-parenthesis-plus notation. Pseudoknots, modified nucleotides, and mismatched strand lengths are rejected.

The new DRfold2, NUMonomer, Zfold, and RNAbpFlow adapters accept one linear RNA chain. They do not infer an oligomer by concatenating chains or independently folding its members. Use an existing multichain evaluator such as AlphaFold 3 or Protenix for oligomer tertiary evaluation.

## Configure secondary scoring

Register a loss under `loss.loss_terms`.

```yaml
loss:
  loss_terms:
    defect:
      _type: rna_ensemble_defect
      backend: nupack
      material: rna
      celsius: 37
      sodium: 1.0
      magnesium: 0.0
      ensemble: stacking
      targets:
        duplex:
          strands: [A, B]
          structure: "(((+)))"
          weight: 1.0
```

This example requires three nucleotides in each protomer. The salt values describe the supported NUPACK RNA parameterization, not an experimental salt correction. Unsupported settings fail rather than being ignored. [NUPACK model specification](https://docs.nupack.org/model/)

Available loss types are:

| Loss | Quantity | Units and direction |
| --- | --- | --- |
| `rna_ensemble_defect` | Weighted normalized complex defect. | Fraction, lower is better. |
| `nupack_ensemble_defect` | The same score with NUPACK explicitly selected. | Fraction, lower is better. |
| `rna_structure_probability` | Equilibrium probability of the exact target structure. | Probability, higher is better. |
| `rna_ensemble_free_energy` | Partition-function ensemble free energy. | kcal/mol. This is not MFE or a target-specific score. |

NUPACK is optional and must be installed under its applicable license. Astrolabe does not download it, accept its terms, or substitute another model when it is unavailable. The adapter uses its public Python utilities. [NUPACK installation](https://docs.nupack.org/start/) · [NUPACK utilities](https://docs.nupack.org/utilities/)

The explicit alternative `backend: viennarna` supports a single strand. Omit `sodium` and `magnesium`, and set `ensemble: vienna` or omit it. ViennaRNA and NUPACK use different thermodynamic models. Their values are not interchangeable validation evidence. [ViennaRNA Python API](https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/api/python.html)

## Configure the phase transition

The example [rna_secondary_to_joint.yaml](../config/rna_secondary_to_joint.yaml) starts with random RNA sequences and random substitutions.

Its default stage uses the `sequence` evaluator and maximizes `-defect`. `sequence` supplies no coordinates or confidence values. It does not load a tertiary model.

The joint stage enters when the accepted sequence satisfies `defect <= rna_secondary.threshold`. Rejected proposals cannot trigger entry. An already qualifying initial sequence can enter at step zero.

Entry evaluates that same accepted sequence using the tertiary evaluator. Astrolabe then establishes its joint score as the comparison baseline. Subsequent proposals use the joint objective. The transition does not reverse automatically.

`active_loss_terms` controls which losses are computed, not merely their weights. Omitting this field preserves the previous behavior. With an explicit list, include every loss needed by that stage's objective, acceptance conditions, stopping conditions, and stage-entry expressions.

The example threshold `0.10` and secondary weight `1.0` illustrate configuration. They are not experimentally calibrated cutoffs. Choose them for the target and scoring model. The example's joint objective uses pLDDT and defect. It does not establish assembly yield or correct interchain geometry.

Run through the normal CLI after configuring the selected dependencies.

```bash
astrolabe run --task rna_secondary_to_joint --dry-run
astrolabe run --task rna_secondary_to_joint --overrides 'out_dir=/path/to/new/run'
```

Keep one shared scientific configuration for a batch. Put only sequence, seed, input assets, and output identity in per-run overlays.

## Select a single-chain predictor

Configure one of `drfold2`, `numonomer`, `zfold`, or `rnabpflow` as an evaluator type.

```yaml
evaluator:
  evaluator_terms:
    tertiary:
      _type: numonomer
      repo_path: /path/to/NUMonomer
      weights_path: /path/to/NUMonomer.pt
      device: cuda:0
      execution_python_bin: /path/to/compatible/python
```

The model owner runs in an isolated interpreter. Provide an interpreter that can import Astrolabe and the selected upstream dependencies. The adapter does not install packages or fetch weights. Its source revision is recorded separately from the installed model assets.

Required backend assets are configured in the same evaluator term:

| Type | Assets and additional settings |
| --- | --- |
| `drfold2` | `repo_path`, `weights_path` pointing to the upstream weight directory, including the experiment checkpoints and `RCLM/epoch_67000`. The checkout must include the folding configurations and the Arena executable, or their paths must be supplied explicitly. |
| `numonomer` | `repo_path` and `weights_path` pointing to a checkpoint file. |
| `zfold` | `repo_path` pointing to structRFM, `weights_path` for Zfold, its `config.json`, `lm_weights_path` for structRFM, and `ss_path`. The default `msa_mode: file` also requires `msa_path`. CPU folding requires the upstream Rosetta dependencies. |
| `rnabpflow` | `repo_path`, `weights_path` pointing to a checkpoint, and one explicit conditioning source: `conditioning_dot_bracket`, `conditioning_path`, or ordered `map_paths`. |

Zfold's supplied MSA must match each requested sequence. For random sequence optimization, explicitly selecting `msa_mode: single_sequence` creates a new query-only alignment for each mutation. This is not equivalent to an rMSA. Its `ss_path` remains explicit conditioning. Neural inference uses the request seed, but the inspected upstream Rosetta folding CLI exposes no seed option; the output metadata records that limitation.

RNAbpFlow conditioning is never silently copied from the secondary optimization target. DRfold2 includes CPU potential-based folding and Arena reconstruction after neural inference. RNAbpFlow retains all requested samples and returns the first, without an invented confidence ranking.

Unavailable pLDDT, pTM, ipTM, and PAE remain unavailable. Do not select a confidence loss that the chosen backend cannot produce.

Each evaluation gets its own directory. `astrolabe_prediction.json` records the sequence, seed, selected structure, auxiliary artifacts, and stage contract.

## CPU and GPU ownership

Each backend has separate CPU preparation, device execution, and CPU finalization functions. Device execution includes required device reductions and completed transfer to host memory. CPU finalization does not receive live CUDA tensors.

On the resource-stage runtime, the adapter returns the existing `EvaluatorStages` and `StagedLoader` contracts. The process that executes the model owns resource admission. The coordinator does not hold a duplicate GPU lease around its request.

Splitting model loading also requires the runtime's resource profile to declare its RAM envelope. Without that declaration, the runtime conservatively keeps loading in one admission scope; the per-request prepare, execute, and finalize boundaries remain distinct.

On an older runtime, the same functions execute serially. This preserves functional compatibility but does not provide separate resource admission, overlap, or stage accounting. The adapter rejects an active resource broker when the runtime lacks the stage contract.

Existing recovery restrictions still apply. Keep `search` and `recovery` unset or null for this multistage CLI workflow: the existing portable search/recovery route supports a single static stage, not this transition. This integration does not introduce a separate RNA checkpoint format or promise replay of an unknown model request.

## Upstream implementations

The adapters target these inspected source revisions.

| Backend | Source |
| --- | --- |
| DRfold2 | [leeyang/DRfold2 at 3667d6a](https://github.com/leeyang/DRfold2/tree/3667d6ac44c5b14cff94c36e0371224acb2cdae5). |
| NUMonomer | [yunda-si/NUMonomer at af38434](https://github.com/yunda-si/NUMonomer/tree/af38434e7ee3f82c6948841cce1ca4d65840742a). |
| Zfold | [structRFM Zfold at f0c1add](https://github.com/heqin-zhu/structRFM/tree/f0c1add19f8dfc786f16bc4abb7aa775dbadddbf/tasks/Zfold). |
| RNAbpFlow | [Bhattacharya-Lab/RNAbpFlow at e8b1c07](https://github.com/Bhattacharya-Lab/RNAbpFlow/tree/e8b1c077f7951d4d6ebe00c851d22c37777dfc9f). |

Adapter tests, real thermodynamic calculations, and real pretrained-model inference are different evidence. Use the integration report for the checks performed in the current environment.
