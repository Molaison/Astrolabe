"""Public component controls and their native adapter spelling.

Normalize each configuration layer before merging it. A task override must not
conflict with a default merely because it uses the public spelling.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy


INFERENCE_FIELDS = {
    "alphafold2": {"recycles": "recycles"},
    "alphafold3": {"recycles": "num_recycles", "samples": "num_samples"},
    "rf3": {"recycles": "n_recycles", "samples": "diffusion_batch_size",
            "diffusion_steps": "num_steps"},
    "chai1": {"recycles": "num_trunk_recycles", "samples": "num_diffn_samples",
              "diffusion_steps": "num_diffn_timesteps"},
    "boltz": {"recycles": "recycling_steps", "samples": "diffusion_samples",
              "diffusion_steps": "sampling_steps"},
    "protenix": {"recycles": "model.N_cycle", "samples": "sample_diffusion.N_sample",
                 "diffusion_steps": "sample_diffusion.N_step"},
    "mol1": {"recycles": "num_cycles", "samples": "num_samples", "diffusion_steps": "num_steps"},
    "opendde": {"recycles": "num_cycles", "samples": "num_samples", "diffusion_steps": "num_steps"},
    "esmfold2": {"recycles": "num_loops", "samples": "num_diffusion_samples",
                 "diffusion_steps": "num_sampling_steps"},
}
_INFERENCE_NAMES = frozenset({"recycles", "samples", "diffusion_steps"})
_ABSENT = object()
_EXECUTION_FIELDS = frozenset({
    "execution", "execution_mode", "execution_python_bin", "execution_cwd",
    "execution_env", "execution_request_timeout", "execution_max_frame_bytes",
    "startup_timeout", "shutdown_timeout", "request_timeout", "python_bin",
})
_PROTENIX_FIELDS = frozenset({
    "_type", "model_name", "dynamic_binding_site", "use_constraints",
    "load_checkpoint_path", "load_checkpoint_dir", "load_strict", "seed", "seeds",
    "num_workers", "use_msa", "dtype", "triangle_multiplicative", "triangle_attention",
    "enable_tf32", "enable_efficient_fusion", "enable_diffusion_shared_vars_cache",
    "msa_pair_as_unpair", "use_template", "use_rna_msa", "use_seeds_in_json",
    "use_deepspeed_evo_attention", "infer_setting", "data", "model",
    "sample_diffusion", "need_atom_confidence", "sorted_by_ranking_score",
}) | _EXECUTION_FIELDS
_PROTENIX_NESTED_FIELDS = {
    "model": {"N_cycle"},
    "data": {"ccd_components_file", "ccd_components_rdkit_mol_file", "num_dl_workers", "msa", "template"},
    "data.msa": {"enable", "enable_rna_msa"},
    "data.template": {"enable"},
    "sample_diffusion": {"N_sample", "N_step",
                         "guidance", "gamma0", "gamma_min", "noise_scale_lambda", "step_scale_eta"},
    "skip_amp": {"confidence_head", "sample_diffusion"},
    "infer_setting": {"chunk_size", "lddt_metrics_chunk_size", "lddt_metrics_sparse_enable",
                      "sample_diffusion_chunk_size"},
}


def _get(options, path):
    value = options
    for key in path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return _ABSENT
        value = value[key]
    return value


def normalize_evaluator_options(kind, options, *, path="evaluator"):
    """Return native options without mutating the caller's public layer."""
    result = deepcopy(dict(options))
    kind = {"mol-1": "mol1"}.get(str(kind).lower(), str(kind).lower())
    names = INFERENCE_FIELDS.get(kind, {})
    for public in _INFERENCE_NAMES & result.keys():
        if public not in names:
            raise ValueError(f"{path}.{public}: {kind} does not expose this inference control")
        native = names[public]
        if native == public:
            continue
        value = result.pop(public)
        prior = _get(result, native)
        if prior is not _ABSENT and (type(prior) is not type(value) or prior != value):
            raise ValueError(f"{path}.{public}: conflicts with {path}.{native}; use only {public}")
        parent = result
        parts = native.split(".")
        for part in parts[:-1]:
            if part in parent and not isinstance(parent[part], Mapping):
                raise ValueError(f"{path}.{part}: expected mapping")
            parent = parent.setdefault(part, {})
        parent[parts[-1]] = value
    return result


def component_ownership_errors(family, kind, options, *, path, runtime=False):
    """Reject fields that Astrolabe or a checkpoint owns, before model loading."""
    errors = []
    kind = str(kind).lower()
    if family == "evaluator" and kind == "protenix":
        allowed = _PROTENIX_FIELDS | ({"dump_dir", "input_json_path", "skip_amp"} if runtime else set())
        for key in options.keys() - allowed:
            errors.append(f"{path}.{key}: not a public Protenix inference control; model presets own training and architecture settings")
        for prefix, keys in _PROTENIX_NESTED_FIELDS.items():
            value = _get(options, prefix)
            if isinstance(value, Mapping):
                for key in value.keys() - keys:
                    errors.append(f"{path}.{prefix}.{key}: owned by the native model preset, not task configuration")
    if family == "evaluator" and kind == "alphafold3":
        for key in ("json_path", "input_dir", "output_dir"):
            if options.get(key) not in (None, "", "None"):
                errors.append(f"{path}.{key}: derived from the current input/output owner; remove this override")
    if family == "mutator" and kind in {"proton_pottsmpnn", "proton-pottsmpnn"}:
        if _get(options, "criteria.max_mutations") not in (_ABSENT, None):
            errors.append(f"{path}.criteria.max_mutations: derived per proposal; configure stages.*.mutation_num")
    return errors
