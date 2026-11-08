# Dynamic joint His-network proposals

This opt-in extension proposes canonical sequences through Proton-PottsMPNN. It does not install a motif, pin a His position, or replace the measured AF3 network score with a virtual geometry score.

For every proposal, `network_proposal` reads the current accepted binder structure and current mutable mask. It scans spatially neighboring allowed sites and compatible fixed His/Arg/Lys partners. Neutral His and donor rotamers are screened for hydrogen-bond distance/angle, sidechain clashes, actual Rosetta H-bond energy, and atomic depth in the isolated binder. The parent structure/sequence remains unchanged. At most 16 sparse role preferences are sent to the native Potts engine, indexed by zero-based binder sequence position.

The native proposal objective is `H / frozen_sdH - existing_AA_reward - b_pair * max(q_pair * role_match)`. The pair reward is capped by the best supported edge and never changes the reported raw Potts Hamiltonian or protonation-state energies. Both roles must match the proposed canonical sequence. Pair positions are mapped through the binder chain's context indices, including when the binder is not the first chain. Fixed residues remain fixed, and step/reference mutation budgets are applied to the full joint assignment.

The solver compares its usual best single-improvement block with every geometry-supported pair block. It attempts pair blocks even if neither single substitution improves. Temperature sampling in this opt-in branch uses only improving assignments, and `network_pair_max_steps` bounds the total number of moves. With the feature disabled or preferences empty, the original solver branch is used.

After the resulting canonical sequence is evaluated by AF3, the existing `ph_core_network` definitions determine N, R, G and U from the actual structure. Virtual rotamer witnesses are logged separately as `ph_network_proposal_audit`, with `observed_network=false`. Only `ph_core_network_audit` records are evidence of observed AF3 networks. Software positive controls are never optimizer parents.

The pilot uses pair weight 6, at most 12 moves, 32 candidate spatial pairs, 12 rotamers per identity/site, a 3–9 A CB distance window, sequence separation at least 3, no sidechain heavy-atom clash below 2 A, and a virtual pair mean depth of at least 3 A. These are engineering proposal settings, not published universal thresholds or physical pH/free-energy units. His-only bias and the actual AF3 N/R/U definitions remain unchanged. No positive result is assumed from feasible virtual geometry.

Set `mutator.mutator_terms.<alias>.network_proposal.enabled: true` and `criteria.network_pair_bias: 6.0`, with `criteria.network_pair_max_steps: 12`, on a centre-free `greedy_energy_block` term. Default `enabled: false` preserves existing behavior. Actual pilot overlays and frozen provenance are retained in the project revision for reproduction.
