# Internal neutral-His network proposals

Set `network_only: true` on a `proton_pottsmpnn` mutator term. This opt-in
Astrolabe adapter profile enables the local Proton solver extension
`network_pair_only`; the default remains false for existing callers.

The adapter examines the current binder on every round. The native sequence
`+` mask fixes the interface; `position_selector: not surface()` selects the
mutable interior. Virtual side-chain geometry generates His–His/Lys/Arg pair
roles. Proton ranks or samples compatible assignments using its neutral Potts
potential, allowing at most one pair and two canonical edits. His uses HIS-S.
There are no preinstalled His residues or permanently selected His sites.
A fixed matching partner can be retained without editing it. A later round
can propose another pair or change an existing pair member to a different role.

This profile supplies no standalone amino-acid reward, pair energy bonus,
whole-chain redesign, or cumulative parent-reference mutation budget. Validated
pair geometry is preferred. If no pair passes the virtual geometry checks,
the adapter still tries the best legal pair. Ranking first minimizes the
distance outside the usual CB interval, then sequence-separation shortfall,
then prefers deeper anchors and proximity to the interval midpoint. CA is a
coarse anchor when CB is absent. Only one positional pair is selected; Proton
chooses its compatible His/His-Lys-Arg roles. Fixed identities remain fixed.
Fallback preferences have quality zero and no validated virtual witness; role
eligibility is independent of quality and adds no energy reward. A canonical
no-op remains possible when no legal changed pair exists. A candidate
may have worse neutral Potts energy than the current sequence: it is a proposal
for the outer structure-and-score acceptance step, not an accepted design.
The adapter independently rejects edits outside a proposed pair before
applying them to sequence segments.

Use these loss terms and stage expressions:

```yaml
# iptm_binder_target: chain_pair_iptm, chain_a: A, chain_b: [B, C, D]
# binder_ptm: binder_ptm, binder_chain: A
# neutral_preference: ph_core_network, metric: release, binder_chain: A
# unpaired_his: ph_core_network, metric: unpaired, binder_chain: A
weighted_score_expression: iptm_binder_target + 0.4 * binder_ptm + neutral_preference
acceptance_gate: neutral_preference > 0 and unpaired_his == 0
```

The current optimization contains binder A and target trimer B/C/D. Target
sequences and binder interface segments are fixed using native `seqs` plus
segments. `iptm_binder_target` averages only A-B, A-C and A-D chain-pair iPTM;
target-target pairs do not enter it. Binder pTM has the user-requested weight
0.4. Interface-motif RMSD is no longer in the current composite score.
`neutral_preference = N * tanh((H_protonated - H_neutral) / nHis)` at default
gap reference 0 and scale 1. The raw Potts energies use the same coordinates
and sequence, with selected network His switched between HIS-P and HIS-S.
`N` is the existing Rosetta hydrogen-bond/burial network quality; it is already
included, so it receives no separate reward. `unpaired_his` counts deeply
buried His without a qualifying hydrogen bond. The definitions and empirical
cutoffs are documented in [ph_core_automatic.md](ph_core_automatic.md).

The gate requires a detected network with a strictly positive neutral-state
preference and zero deeply buried unpaired His. Strict greedy acceptance also
requires the total score to improve. The normal initial-state creation bypasses
the gate, so a zero-His root can start. Geometry/quality cutoffs, fallback
ranking, the release-term coefficient and proposal temperature are engineering choices,
not experimentally calibrated pH thresholds.

This is a neutral-versus-protonated computational preference. Neither the gate
nor complex AF3 measures pH-specific affinity, proves pH 7.4 binding, or proves
acidic release. CPU qualification tests proposals and acceptance logic on two
zero-His parents; it does not establish post-mutation AF3 network formation.
Model weights and raw Potts/state energy definitions are
unchanged. The opt-in solver diff is recorded in
`patches/vendor/proton_pottsmpnn/04_network_only_proposals.patch`, followed by
`05_best_effort_pair_roles.patch`.
