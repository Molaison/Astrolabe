# Proton-PottsMPNN installation and supported boundary

Astrolabe includes the selected Proton source at upstream commit
`09682abfa7d20e0abcdeea0490b7a4b1c190aee3`, plus five declared patches.
`vendor/proton_pottsmpnn/SOURCE_PROVENANCE.json` records each source file,
its upstream Git blob, SHA256 and size, and the patch order and working
directories. Reversing the patches reconstructs the pinned upstream files.
The patches preserve the design mask and add the existing histidine/network
proposal behavior. This is a patched adapter, not pristine upstream execution.

The public source archive contains these sources and patches. The Python wheel
contains the Astrolabe interfaces; it does not contain vendor code. Run model
work from a source checkout/archive with its vendor directory, or point the
configuration at that directory explicitly.

## External environment and assets

Use a separate Python 3.12 environment with the dependencies declared by the
pinned Proton `foundry/pyproject.toml`, MPNN package and
`requirements-extra.txt`. Follow the pinned upstream installation instructions
in a dedicated environment. Do not install that stack in Astrolabe's main
Python environment. The daemon selects `foundry/src` and
`foundry/models/mpnn/src` from the configured source directory.

Model assets are acquired separately and are excluded from Git and public
distributions. The manifest's `runtime_assets` list gives immutable download
URLs, sizes and SHA256 values for the v6 checkpoint and ten EV6 labeller folds.
Verify those hashes and place each file at its recorded path relative to the
Proton source root. The checkpoint may instead be placed under an external
`GH_WEIGHT_PROTON_POTTSMPNN` directory with the same subdirectory structure.
EV6 folds remain under the selected Proton source tree.

Supply a working HBPLUS executable under its own installation terms. The
current Astrolabe adapter uses the labeller and therefore requires HBPLUS.
Configure the existing external-environment interface:

```bash
export GH_PROTON_POTTSMPNN_PYTHON=/absolute/proton-env/bin/python
export GH_PROTON_POTTSMPNN_REPO=/absolute/Astrolabe/vendor/proton_pottsmpnn
export GH_PROTON_POTTSMPNN_HBPLUS=/absolute/hbplus
export GH_WEIGHT_PROTON_POTTSMPNN=/absolute/proton-checkpoints
```

The checkpoint suffix is
`potts_v6_afdb_edge_his0.3_acid0.06/epoch-0125.ckpt`. In a container, these
paths and the complete interpreter environment must actually be visible and
compatible inside that container. Host paths alone do not establish this.
Source inventory validation does not require weights; a passing source-only
installation is not model readiness.

## Use and interpretation

Select `proton_pottsmpnn` or `Proton-PottsMPNN` through the normal mutator
configuration; shared defaults live in `mutator.ProtonPottsMPNN`. Use
`astrolabe run --task <task> --dry-run` to check the selected configuration.
See [pH design](ph_sensitive_design.md), [automatic core
design](ph_core_automatic.md) and [network proposals](ph_joint_network_proposals.md).
Standalone `astrolabe posthoc proton-states` and `proton-parity` have their own
`--help`; parity arms require their exact source trees and inputs.

This integration supports the existing protein-only v6 adapter and its pH
loss/posthoc interfaces. It preserves masks, joint proposals, budgets and RNG
behavior. It does not add vdM or independent Rosetta pH analysis tools.
`HIS-S` is neutral and `HIS-P` protonated. Potts state scores are uncalibrated
model proxies, not numerical pH, pKa or binding free energies. The existing
`network_only` mode uses centre-free neutral-pair proposals with selectivity
disabled. Installation and CPU interface tests do not certify GPU throughput
or experimental pH selectivity.
