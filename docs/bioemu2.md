# BioEmu2 evaluator

The `bioemu2` evaluator runs an independently installed BioEmu2 Python in a
subprocess. It accepts protein-only Astrolabe inputs and uses a local,
query-only A3M per chain: no sequence is submitted to a remote MSA service.
It does not use Astrolabe's supplied MSAs, templates or restraints. Protein
modifications, cyclic chains and explicit bonds are rejected.

Example evaluator configuration (set paths for your installation):

```yaml
# Set this to a persistent directory; Astrolabe otherwise removes its
# automatically managed raw results directory after a successful run.
raw_model_results_dir: /path/to/persistent/raw_model_results
evaluator:
  evaluator_terms:
    BioEmu2:
      _type: bioemu2
      python_bin: /path/to/bioemu2/.venv/bin/python
      cache_dir: /path/to/bioemu2/huggingface-cache
      mode: equilibrium
      num_samples: 8
      batch_size: 1
      sampler: edm
      # Optional: default uses the first Astrolabe input seed.
      # seed: 20261020
stages:
  default:
    evaluator: BioEmu2
    # Specify oligo, mutators and a compatible structure-based score as usual.
```

`mode: structure` requests BioEmu2 structure-mode sampling instead.
`mode: propagation` requires `initial_cif`, and optionally accepts
`lag_time_ns` (default 1.0) and `propagation_steps` (default 1). It starts
`num_samples` replicas from that same structure and saves their final endpoints;
intermediate steps are not saved. The CIF must contain every atom expected by
BioEmu2 with matching chain IDs, residues and sequence. A mutated candidate
does not automatically acquire new side chains or a new initial structure;
mismatches fail rather than silently propagating the wrong sequence.

Every evaluation creates a separate directory under
`raw_model_results_dir/bioemu2/<alias>/`. It contains the request, worker log,
query MSAs, `samples.xtc`, `topology.cif`, `physicality.json`, `result.json`,
and `representative.cif`. Astrolabe's current `complex_paths` contract passes
only the **first frame that passes BioEmu2 physicality** to structure losses;
it does not average scores over the ensemble or select the best score. If no
frame passes, the evaluator fails. Use the saved trajectory for separate
ensemble analysis. BioEmu2 does not produce pLDDT, pTM, ipTM, PAE or PDE;
confidence-dependent losses and selectors must not be used for this evaluator.

Weights must already be available in `cache_dir`; the worker defaults to
`HF_HUB_OFFLINE=1`. The separately installed environment and model weights are
not bundled with Astrolabe. Configure device, precision and matmul precision
with `device`, `precision` and `matmul_precision` if needed; defaults are
`cuda`, `bf16-mixed` and `high`. Propagation is model-based conformational
propagation, not continuous molecular dynamics or proof of equilibrium,
binding or a folding mechanism.
