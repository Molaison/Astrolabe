# Binder-only pH network optimization

> The original `interface_motif_rmsd` / `interface_motif_rmsd_norm` implementation
> and its original tests are included in this Proton/pH port. This recipe uses
> the existing fixed-segment syntax and those geometry metrics; unified CPU/SIF
> validation of the current candidate remains with the release owner. Code
> availability does not establish a scientifically qualified binder workflow.

Run the existing automatic His and donor network proposals on the binder alone.
Define the interface once using the original bound reference complex. Preserve
that interface's sequence with the existing `seqs` fixed segment syntax:

```yaml
seqs:
  A: AAA +KLE AAA +FQ AAA
types:
  A: protein
stages:
  default:
    oligo: [A]
```

This is a syntax example, not a design sequence. Each whitespace-delimited
`+LITERAL` segment is fixed by `Protomer.mutable_mask`, which the Proton mutator
already propagates to its native `fixed_positions`. Unprefixed segments remain
mutable. No additional protection parameter or fixed His site is needed.

Use `position_selector: not surface()` to restrict proposals to the internal
region. The frozen interface mask remains in force independently of the
selector. His is introduced by the existing proposal biases during optimization;
the input parent can contain zero His.

## Frozen interface geometry

`reference_interface_positions(reference_complex, reference_binder_chain,
reference_target_chains, contact_cutoff)` returns zero-based binder sequence
positions with any heavy atom within the cutoff of any selected reference target
heavy atom. Calculate these positions before the run and encode them in `seqs`.
The qualification overlays use a 4.5 angstrom cutoff, inherited from the previous
interface selector. This is an engineering definition of this design's protected
interface, not an experimentally established pH threshold.

The `interface_motif_rmsd` loss compares N/CA/C/O atoms at all those positions
with one joint optimal rigid body fit. It requires a complete binder of the same
length, unchanged interface amino acids and matching backbone atom coverage.
It permits substitutions outside the frozen interface. The target chains exist
only in the reference file; they are not required in the prediction.

```yaml
loss:
  loss_terms:
    motif:
      _type: interface_motif_rmsd_norm
      reference_complex: /path/to/original_bound_complex.cif
      binder_chain: A
      reference_binder_chain: A
      reference_target_chains: [B, C, D]
      contact_cutoff: 4.5
      best: 0.0
      worst: 2.0
```

The normalized score is `M = clip((worst - RMSD)/(worst - best), 0, 1)`;
higher is better. The qualification uses `best=0`, `worst=2` angstrom and
coefficient 1.0. These are initial engineering settings to calibrate, not paper
derived cutoffs. Record the raw RMSD as a separate zero-weight term for auditing.
This score supplies geometric guidance; the native `+` mask fixes sequence.

## Composite score

With only one chain, use binder pTM for fold confidence. The qualification uses

```text
Q_apo = binder_ptm - 0.3*sigmoid((binder_rog-globin_rog)/5)
                   - 0.6*min(binder_loop_frac, 1)
F = Q_apo + 1.0*M + 0.2*N + 0.1*R - 0.1*U
```

N, R and U retain the previous neutral network, two-state release proxy and
unpaired buried His definitions. Native strict greedy acceptance still compares
the complete F against the current accepted state. No new acceptance gate is
introduced. Preserve the existing His bias, joint proposal bias and mutation
budget when qualifying this input change. A single-chain confidence/geometry
score and an apo protonation-state energy difference do not establish binding
affinity or experimental pH-dependent release.
