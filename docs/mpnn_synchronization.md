# MPNN synchronization

The `ProteinMPNN` and `LigandMPNN` mutators accept an opt-in synchronization policy. The default is `legacy_order_conditioned_marginals`, which retains the existing sample-time distributions, aggregation and random sampling. Existing `MultistateProteinMPNN` and `FarApartTiedMultitargetLigandMPNN` aliases compile to that legacy policy with a far-apart layout. The aliases do not mean the same algorithm as the opt-in joint policy below.

```yaml
mutator:
  mutator_terms:
    design:
      _type: ProteinMPNN
      context_layout: separate
      coupling: probability_mean
      context_weights: {AB: 1.0, AC: 1.0}
```

`probability_mean` calls the existing MPNN `score(..., use_sequence=True)` once per context, with deterministic canonical decoding ranks and the accepted sequence from the evaluated structures. It applies the runner's existing amino-acid bias, omissions and temperature before averaging probabilities. Shared segment positions are mapped to a canonical variable, weighted across contexts, and sampled once by `SegmentManager`. This is a parallel, order-conditioned teacher-forcing proposal: different contexts have different structural and target conditioning, and each shared position sees the earlier reference residues in the canonical decoding order. It is not a sample from the model's joint distribution. The runtime records the distribution semantics and a fingerprint of the shared reference, masks and decoding orders.

The adapter checks physical chain order, alphabet, sequence, design masks, shared decoding order and normalized finite probabilities. Missing capability or incompatible conditioning fails. Negative context weights are not supported by probability averaging. This mode currently requires one physical copy of each shared segment position per context, one reference sample, and no supplemental-structure or transition-consistency blending. A structure whose sequence is not the accepted sequence fails instead of being silently scored as that sequence.

```yaml
mutator:
  mutator_terms:
    design:
      _type: ProteinMPNN
      context_layout: far_apart
      coupling: native_tie
```

`native_tie` uses the existing far-apart model call with tied positions and takes one native joint decoded sequence. It does not resample each position from model marginals; the existing mutation budget can select a subset of that decoded sequence. Signed state betas are supported only by the native tie path. The translated combined structure is an internal model input and is not an evaluated physical complex. Native tie currently uses the existing two-context ProteinMPNN adapter. LigandMPNN far-apart mode is rejected because that adapter does not preserve ligand atoms.

The original aliases retain their order-conditioned marginal proposal and RNG behavior; migrating one to `native_tie` changes proposal semantics. Tests use CPU models with random weights and mock backends; no trained-weight or GPU equivalence is implied.

Both MCMC and MPNN builders also accept an optional `proposal_policy` on the mutator term, for example:

```yaml
proposal_policy:
  allowed_residues: [C, D]
  disallowed_residues: []
  bias: {C: 0.5}
  temperature: 1.2
  max_mutations: 1
```

For mutable positions this applies `p(aa)^(1/temperature) * exp(bias(aa)/temperature)` to the backend's probability vector, masks disallowed residues, then normalizes. Fixed/selector-excluded positions are left untouched. The policy's temperature and bias are a post-backend proposal transform, distinct from the MPNN model's native logit temperature and bias. The shared mutation budget is applied after candidate generation and combines with an existing MCMC cap. Unconfigured policy leaves legacy numerical and RNG paths unchanged. Native joint decoding permits the shared budget but rejects probability transforms because post-processing marginals would change joint decoding semantics.
