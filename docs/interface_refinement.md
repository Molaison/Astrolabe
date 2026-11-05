# Binder interface refinement

After installing the AF3, LigandMPNN and physics dependencies, run:

```bash
astrolabe refine interface \
  --input complex.pdb \
  --target-homology target_af3.json \
  --out-dir results
```

`--input` also accepts a directory of PDB/CIF complexes. The longer protein
matching the target homology is identified automatically; ambiguous inputs need
the optional chain overrides. The task starts one private AF3 service for the
batch, runs CPU proposal/scoring workers and drains its service before exiting.
An explicit `--service-socket` reuses a service without taking ownership of it.

The base configuration holds execution defaults. `config/binder_interface_beam.yaml`
holds the historical two-mutation distance-based configuration; the native
interface task in `config/interface.yaml` overrides this with four mutations per
step and at most four total mutations relative to its unmutated root. The binder
is mutable, the target remains fixed, and the five objectives, bounded beam search
and strict pose gates are unchanged. Cases require no YAML. The current default is
one fresh AF3 root and six proposed children per input.
Native coordinates, complete confidence/PAE, scores, rejected candidates, quota
records and the final beam remain in the output directory.

The interface task uses the existing `interface(mode='buried_sasa',
buried_sasa_cutoff=10.0)` position selector. On each parent's predicted complex it
recalculates per-residue absolute buried-SASA deltas using whole-residue SASA:
SASA of the binder chain alone minus SASA of that chain in the pair, strictly
greater than 10 Å². Each child may change only positions selected from its
parent's predicted complex. The inherited
fallback-to-all-mutable behavior remains disabled, so an empty selection cannot
broaden the mutation domain.

Add `--budget 100` to allow at most 100 AF3 calls **per input complex**, including
its unmutated root. A directory of three complexes therefore has a maximum of
300 calls. This single option sets both the beam evaluation limit and its AF3
quota, and allows multiple generations instead of the default single generation.
The proposal count per parent, beam width, four-mutation limits and strict pose
gates keep their task defaults. Search can stop below the budget if no feasible or
distinct candidates remain. `--budget 1` evaluates only the root.

`--prepare-only` validates/prepares inputs without loading models. `--resume`
verifies the original inputs and contract and preserves completed predictions;
a fully completed run does not start AF3 again. Use the original software and
environment for resuming older outputs whose contracts were frozen before an
installation upgrade.
Repeat the same `--budget` value when resuming a budgeted run; omitting or changing
it is rejected before any predictions. To run a different budget, use a new
output directory.

## Machine environment

A normal Python installation uses its current environment. An installation
using Apptainer can put its engine, image, asset mounts and dependency environment
in `config/env/runtime_local.json`, using `runtime.example.json` as a starting
point. This is an installation setting, independent of a case or project.
Both the installed console entry point and `python -m astrolabe` select that
runtime before importing the CLI dependencies. The checkout launcher preserves
its existing interpreter selection for commands without a matching profile.

The interface command preserves an assigned `CUDA_VISIBLE_DEVICES`, including
an explicitly empty restriction. When no device is assigned, it uses the existing
GPU selector to choose the device with most free memory. GPU selection is not a
reservation; shared production batches should use the site's scheduler. Help,
preparation and external service clients hide CUDA. AF3 runs on GPU while MPNN
and physics workers use CPU.
