# RNA secondary-to-tertiary design

Astrolabe can optimize RNA sequence scores before starting a structure predictor. The same stage mechanism also supports a joint objective that retains the secondary score.

## Scientific scope

`rna_ensemble_defect` measures normalized complex ensemble defect. It is the expected fraction of nucleotides whose pairing state differs from a specified target. Lower values are better. An unpaired target nucleotide contributes its unpaired-state probability.

NUPACK's `defect` already returns a normalized value. Astrolabe does not divide it by sequence length again. This score is conditional on the specified complex. It is not test-tube yield and does not include off-target complexes or strand concentrations. [NUPACK definitions](https://docs.nupack.org/definitions/)

Targets preserve strand order and multiplicity. `strands: [A, A]` evaluates two physical copies of the current sequence of protomer A.

For targets with weights \(w_j\), Astrolabe returns \(\sum_j w_j s_j / \sum_j w_j\). Weights must be nonnegative and their total must be positive. The input uses dot-parenthesis-plus notation. Pseudoknots, modified nucleotides, and mismatched strand lengths are rejected.

All four adapters retain their single-chain path. NUMonomer and RNAbpFlow additionally offer **opt-in experimental chain-aware joint inference**, described below. These adaptations change the inputs to monomer-trained weights; they are not native, validated multimer models. DRfold2 and Zfold still reject oligomers with an explanation of their unresolved topology assumptions. No adapter treats ordinary sequence concatenation or independently folded members as oligomer prediction.

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

## Select a predictor

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

## Experimental multichain inference

The default is `multichain_mode: reject`. Enable `experimental` only for NUMonomer or RNAbpFlow:

```yaml
evaluator:
  evaluator_terms:
    tertiary:
      _type: numonomer
      multichain_mode: experimental
      repo_path: /path/to/NUMonomer
      weights_path: /path/to/NUMonomer.pt
      device: cuda:0
      execution_python_bin: /path/to/compatible/python
```

Use Astrolabe's normal oligomer specification, for example `oligo: [AAB]`, to request physical copies of protomers A, A, B. Equal sequences are not deduplicated. The evaluator passes an ordered list of chains, validates every chain, and preserves their distinct output IDs. This experimental interface supports linear, unmodified A/C/G/U RNA with unique single-character chain IDs; explicit interchain covalent bonds and circular assemblies are rejected.

| Backend | Multichain implementation and limitation |
| --- | --- |
| **NUMonomer** | Native positional encoding within each chain; a non-parameter zero row supplies the cross-chain positional contribution to both structural attention and confidence. All cross-chain content/geometric attention remains active in one joint prediction. Checkpoint parameter shapes are unchanged. The native per-residue reconstruction does not introduce a bond across chain boundaries. It also does not enforce chemical geometry. |
| **RNAbpFlow** | Chain-local node positions; cross-chain sinusoidal vectors zeroed **before** the existing relative-position projection (its learned bias remains). One joint flow sampler, full interchain attention, and per-residue atom reconstruction. Explicit conditioning must connect the chain graph. |
| **DRfold2** | Not enabled. Chain handling must change both RCLM and structural positional encodings. PotentialFold additionally treats consecutive packed residues as bonded in its O3′–P springs, local weighting, and clash exclusions. Identical-copy initialization also needs a scientific design decision. Arena can handle distinct chains, but cannot repair an incorrectly specified upstream polymer by relabelling its output. |
| **Zfold** | Not enabled. LM contexts/crops and relative-position features lack chain boundaries. The current Rosetta initializer builds one covalently connected polymer; enabling jump movement does not create disconnected strand chemistry or interchain jumps. Correct support requires joint geometry plus a topology-aware, jointly restrained assembly/refinement path. |

These DRfold2/Zfold limitations are properties of the **released full pipelines**, not a claim that their neural attention mathematically forbids multimers. A research fork is plausible, but simply relaxing the parser would produce the wrong modeled system. Concrete upstream sites: [DRfold2 positional inputs](https://github.com/leeyang/DRfold2/blob/3667d6ac44c5b14cff94c36e0371224acb2cdae5/cfg_95/EvoMSA2XYZ.py#L91-L122), [PotentialFold bond energy](https://github.com/leeyang/DRfold2/blob/3667d6ac44c5b14cff94c36e0371224acb2cdae5/PotentialFold/Optimization.py#L348-L354), [Zfold relative positions](https://github.com/heqin-zhu/structRFM/blob/f0c1add19f8dfc786f16bc4abb7aa775dbadddbf/tasks/Zfold/src/Zfold/network/modules.py#L531-L556), and [Rosetta initialization](https://github.com/heqin-zhu/structRFM/blob/f0c1add19f8dfc786f16bc4abb7aa775dbadddbf/tasks/Zfold/src/Zfold/util/utils_ros.py#L107-L144).

For RNAbpFlow, a minimal **illustrative**, not benchmarked, four-base-per-strand duplex uses:

```yaml
evaluator:
  evaluator_terms:
    tertiary:
      _type: rnabpflow
      multichain_mode: experimental
      repo_path: /path/to/RNAbpFlow
      weights_path: /path/to/RNA3DB.ckpt
      conditioning_dot_bracket: "((((+))))"
      samples_per_sequence: 1
      device: cuda:0
      execution_python_bin: /path/to/compatible/python
```

The `+`-separated strand lengths must match the ordered physical chains. Alternatively supply explicit finite global-order `conditioning_path`/`map_paths`. Positive interchain entries, across the supplied channels, must form a connected graph over all chains; disconnected conditioning is rejected. This is an adapter scope restriction, not a claim that disconnected RNA complexes are physically impossible. The secondary optimization target is never copied implicitly, even with `use_target_dot_bracket`. Supplied interchain contacts are **conditioning**, not independently predicted interface evidence. [RNAbpFlow's paper describes single-chain RNA monomers](https://www.nature.com/articles/s41592-026-03128-4).

Both adaptations serialize slices of the **same joint coordinates** into one multichain structure, without independent folding, fitting, or docking after prediction. Residue numbering restarts within each chain. Metadata records physical chain layout, experimental method, conditioning where applicable, and unvalidated accuracy. Scoped positional hooks are removed even after inference failure; single-chain requests do not install them.

NUMonomer's pLDDT is still the upstream monomer score, not an interface score. If `clamp_plddt` leaves an unscored suffix, the adapter retains the prefix in metadata and does not publish a fabricated full-length confidence array. Full-length confidence remains one assembly array per sample; per-protomer scores retain repeated physical occurrences. RNAbpFlow provides no pLDDT, pTM, ipTM or PAE: do not reuse the example's pLDDT-based joint objective with that backend. Use an available coordinate-based objective with the secondary score instead.

The existing secondary-to-joint stage logic is unchanged: the accepted sequence crosses the configured secondary threshold, then that same sequence is re-evaluated by the selected tertiary backend to establish the joint baseline. Opting into experimental multichain inference does not change the scientific meaning of NUPACK complex defect or qualify the model's assembly accuracy.

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
