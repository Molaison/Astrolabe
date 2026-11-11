"""CPU-only contracts for pinned Molence base/Constraint selection."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess


SOURCE_COMMIT = "9e85ad16b3d7031dd92c34d99f895ee65547038b"
BASE_MODEL_NAME = "mol-1-base"
MODEL_NAME = "mol-1-constraint"
CONFIG_KEYS = frozenset({
    "python_bin", "repo_path", "mol_root_dir", "load_checkpoint_path", "reference_assets",
    "model_name", "num_cycles", "num_steps", "num_samples", "num_workers", "dtype",
    "use_msa", "need_atom_confidence", "triangle_multiplicative", "triangle_attention",
    "enable_fusion", "enable_tf32", "deterministic", "device", "timeout_seconds",
    "execution_mode", "constraints", "_type",
    "extra_python_paths", "base_checkpoint_path", "constraint_checkpoint_path",
    "dynamic_binding_site",
})
REFERENCE_FILES = {
    "ccd_components_file": "common/components.cif",
    "ccd_components_rdkit_mol_file": "common/components.cif.rdkit_mol.pkl",
    "pdb_cluster_file": "common/clusters-by-entity-40.txt",
    "obsolete_release_data_csv": "common/obsolete_release_date.csv",
}


def configuration_errors(settings, *, allow_unresolved=False):
    errors = [f"{key}: unsupported Mol-1 field" for key in sorted(set(settings) - CONFIG_KEYS)]
    for key, value in settings.items():
        if allow_unresolved and isinstance(value, str) and "${" in value:
            continue
        if key in {"num_cycles", "num_steps", "num_samples", "num_workers"}:
            if type(value) is not int or value < (0 if key == "num_workers" else 1):
                errors.append(f"{key}: expected {'nonnegative' if key == 'num_workers' else 'positive'} integer")
        elif key in {"use_msa", "need_atom_confidence", "enable_fusion", "enable_tf32", "deterministic", "dynamic_binding_site"}:
            if type(value) is not bool or key == "need_atom_confidence" and not value:
                errors.append(f"{key}: expected boolean" + (" true (identity-safe confidence required)" if key == "need_atom_confidence" else ""))
        elif key == "model_name" and value not in ("auto", BASE_MODEL_NAME, MODEL_NAME):
            errors.append("model_name: expected auto/mol-1-base/mol-1-constraint")
        elif key == "execution_mode" and value != "daemon":
            errors.append("execution_mode: requires the existing native daemon owner")
        elif key == "dtype" and value not in {"fp32", "fp16", "bf16"}:
            errors.append("dtype: expected fp32/fp16/bf16")
        elif key == "timeout_seconds" and value is not None:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                errors.append("timeout_seconds: expected positive finite number or null")
        elif key == "reference_assets":
            if not isinstance(value, Mapping) or set(value) - set(REFERENCE_FILES):
                errors.append("reference_assets: unsupported reference keys")
        elif key == "extra_python_paths" and value is not None:
            if (not isinstance(value, (str, list, tuple))
                    or isinstance(value, (list, tuple)) and any(not isinstance(path, str) or not path.strip() for path in value)):
                errors.append("extra_python_paths: expected path-separated string or list of nonempty paths")
        elif key == "constraints" and value is not None:
            try:
                active_constraints(value)
            except ValueError as error:
                errors.append(f"constraints: {error}")
    return errors


def active_constraints(raw):
    """Keep active primitives; recognize official null/empty forms, reject unknown fields."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping) or set(raw) - {"site", "contact", "exclusion"}:
        raise ValueError("Mol-1 constraints only support site/contact/exclusion")
    result = {}
    for kind, value in raw.items():
        if value is None:
            continue
        allowed = {"pairs", "token_pairs"} if kind == "contact" else {"source_chains", "partner_chains", "residue_ids"}
        if not isinstance(value, Mapping) or set(value) - allowed:
            raise ValueError(f"Mol-1 {kind} contains unsupported fields")
        if kind == "contact":
            if len(value) > 1:
                raise ValueError("Mol-1 contact cannot mix pairs and token_pairs")
            items = value.get("token_pairs", value.get("pairs", []))
        else:
            if value and "residue_ids" not in value:
                raise ValueError(f"Mol-1 {kind}.residue_ids is missing; use [] for none")
            items = value.get("residue_ids", [])
        if not isinstance(items, (list, tuple)):
            raise ValueError(f"Mol-1 {kind} residues/pairs must be a list")
        if items:
            result[kind] = deepcopy(dict(value))
    return result


def selected_model(settings, job=None, *, constraint_hint=False):
    """A hint may choose initial residency; every actual job is checked again."""
    configured = active_constraints(settings.get("constraints"))
    requested = active_constraints(job.get("constraints")) if job is not None else {}
    if (settings.get("model_name", "auto") == MODEL_NAME
            or settings.get("dynamic_binding_site", False)
            or (job is None and constraint_hint) or configured or requested):
        return MODEL_NAME
    return BASE_MODEL_NAME


def checkpoint_paths(settings, *, model_name=None, allow_unresolved=False):
    """Config freezes both switchable assets; a native request binds only its selection."""
    mode = settings.get("model_name", "auto")
    if allow_unresolved and isinstance(mode, str) and "${" in mode:
        return {} # The intentional unresolved dry-run cannot select a mode yet.
    if mode not in ("auto", BASE_MODEL_NAME, MODEL_NAME):
        raise ValueError("Mol-1 model_name must be auto/mol-1-base/mol-1-constraint")
    if model_name not in (None, BASE_MODEL_NAME, MODEL_NAME):
        raise ValueError("Mol-1 native selection must be mol-1-base or mol-1-constraint")
    names = (model_name,) if model_name else ((MODEL_NAME,) if mode == MODEL_NAME else (BASE_MODEL_NAME, MODEL_NAME))
    legacy = settings.get("load_checkpoint_path")
    legacy_model = mode
    if legacy and mode == "auto":
        if allow_unresolved and isinstance(legacy, str) and "${" in legacy:
            return {} # Resolve the legacy filename before assigning its model.
        legacy_model = Path(legacy).stem
        if legacy_model not in (BASE_MODEL_NAME, MODEL_NAME):
            raise ValueError("Mol-1 auto load_checkpoint_path must name mol-1-base.pt or mol-1-constraint.pt")
    paths = {}
    for name in names:
        field = "base_checkpoint_path" if name == BASE_MODEL_NAME else "constraint_checkpoint_path"
        raw = legacy if legacy and legacy_model == name else settings.get(field)
        if not raw:
            raise ValueError(f"Mol-1 requires {field}; no fallback to another model's weights")
        path = Path(raw).expanduser()
        if not (allow_unresolved and "${" in str(path)) and path.name != name + ".pt":
            raise ValueError(f"Mol-1 {name} requires {name}.pt, never another model's weights")
        paths[name] = path
    return paths


def local_assets(settings, *, model_name=None):
    """Enumerate config-wide or selected native assets, without imports/downloads."""
    source = Path(settings["repo_path"]).expanduser().resolve(strict=True)
    for name in ("mol/__init__.py", "mol/constraint/inference.py", "mol/constraint/model.py",
                 "mol/constraint/checkpoint.py", "mol/constraint/schema.py", "mol/runner/batch_inference.py"):
        if not (source / name).is_file():
            raise FileNotFoundError(f"Mol-1 source missing {name}")
    top = subprocess.check_output(["git", "-C", str(source), "rev-parse", "--show-toplevel"], text=True).strip()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if Path(top).resolve() != source or head != SOURCE_COMMIT:
        raise ValueError(f"Mol-1 requires source checkout {SOURCE_COMMIT} at {source}")
    root = Path(settings["mol_root_dir"]).expanduser().resolve(strict=True)
    overrides = settings.get("reference_assets") or {}
    if not isinstance(overrides, Mapping) or set(overrides) - set(REFERENCE_FILES):
        raise ValueError("Mol-1 reference_assets contains unsupported fields")
    references = {}
    for key, relative in REFERENCE_FILES.items():
        default = (root / relative).resolve(strict=True)
        if key in overrides:
            value = overrides[key]
            if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
                raise ValueError(f"Mol-1 reference_assets.{key} must name the root's default file")
            supplied = Path(value).expanduser().resolve(strict=True)
            if supplied != default:
                # Official CCD imports cache data_configs paths before runner
                # overrides apply. Bind identity only to what is actually read.
                raise ValueError(
                    f"Mol-1 does not support replacing reference_assets.{key}; "
                    f"expected root default {default}, got {supplied}"
                )
        references[key] = str(default)
    checkpoints = [path.resolve(strict=True) for path in checkpoint_paths(settings, model_name=model_name).values()]
    adapter = Path(__file__).resolve().parent
    paths = [*checkpoints, *map(Path, references.values()), *sorted((source / "mol").rglob("*.py")),
             adapter / "mol1.py", adapter / "mol1_contract.py",
             adapter / "wrappers/mol1.py", adapter / "wrappers/mol1_daemon.py"]
    if any(not path.is_file() for path in paths):
        raise ValueError("Mol-1 assets must be local files")
    return paths, references


def explicit_seeds(values):
    if (not isinstance(values, (list, tuple)) or not values
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in values)
            or len(set(values)) != len(values)):
        raise ValueError("Mol-1 requires explicit, unique integer seeds in [0, 2**32-1]")
    return list(values)


def constraints_for_chains(raw, lengths):
    """Validate the shipped inline JSON contract without coercion or field loss."""
    result = active_constraints(raw)

    def residue(value):
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or not isinstance(value[0], str) or value[0] not in lengths
                or type(value[1]) is not int or not 1 <= value[1] <= lengths[value[0]]):
            raise ValueError(f"Invalid Mol-1 [chain_id, 1-based label_seq_id]: {value!r}")
        return list(value)

    for kind in ("site", "exclusion"):
        value = result.get(kind)
        if value is None:
            continue
        keys = {"source_chains", "partner_chains", "residue_ids"}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError(f"Mol-1 {kind} requires exactly {sorted(keys)}")
        for role in ("source_chains", "partner_chains"):
            chains = value[role]
            if (not isinstance(chains, (list, tuple)) or not chains
                    or any(not isinstance(c, str) or c not in lengths for c in chains)
                    or len(set(chains)) != len(chains)):
                raise ValueError(f"Mol-1 {kind}.{role} requires unique present chains")
        if set(value["source_chains"]) & set(value["partner_chains"]):
            raise ValueError(f"Mol-1 {kind} source/partner overlap")
        ids = value["residue_ids"]
        if not isinstance(ids, (list, tuple)):
            raise ValueError(f"Mol-1 {kind}.residue_ids must be a list")
        ids = [residue(item) for item in ids]
        if any(cid not in value["source_chains"] for cid, _ in ids):
            raise ValueError(f"Mol-1 {kind} residue outside source chains")
        result[kind] = {**value, "residue_ids": ids}
    value = result.get("contact")
    if value is not None:
        if not isinstance(value, Mapping) or len(value) != 1 or not set(value) <= {"pairs", "token_pairs"}:
            raise ValueError("Mol-1 contact requires exactly pairs OR token_pairs; distances are not configurable")
        key = next(iter(value))
        pairs = value[key]
        if not isinstance(pairs, (list, tuple)):
            raise ValueError("Mol-1 contact pairs must be a list")
        translated = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Mol-1 contact pair requires two endpoints")
            if key == "pairs":
                endpoints = [residue(item) for item in pair]
            else:
                if any(type(i) is not int or not 0 <= i < sum(lengths.values()) for i in pair):
                    raise ValueError("Mol-1 contact token_pairs must be zero-based protein token indices")
                endpoints = list(pair)
            if endpoints[0] == endpoints[1]:
                raise ValueError("Mol-1 CONTACT endpoints must differ")
            translated.append(endpoints)
        result["contact"] = {key: translated}
    return result


def output_samples(output_dir, job, count):
    """Require the exact requested seed/sample file set, in explicit input order."""
    root = Path(output_dir)
    seeds = explicit_seeds(job.get("modelSeeds"))
    job_dir = root / job["name"]
    if {p.name for p in job_dir.glob("seed_*")} != {f"seed_{seed}" for seed in seeds}:
        raise ValueError("Mol-1 output seed set differs from the explicit request")
    samples = []
    for seed in seeds:
        directory = job_dir / f"seed_{seed}" / "predictions"
        expected = set()
        optional = set()
        for sample in range(count):
            prefix = job["name"]
            summary = directory / f"{prefix}_summary_confidence_sample_{sample}.json"
            full = directory / f"{prefix}_full_data_sample_{sample}.json"
            cif = directory / f"{prefix}_sample_{sample}.cif"
            expected.update((summary.name, full.name, cif.name))
            # Native save_structure_cif may also emit an explicitly derived
            # unresolved-atom-filtered structure. It is not another sample.
            optional.add(f"{prefix}_sample_{sample}_wounresol.cif")
            if any(not path.is_file() for path in (summary, full, cif)):
                raise ValueError(f"Mol-1 missing output seed={seed} sample={sample}")
            samples.append((seed, sample, summary, full, cif))
        actual = {p.name for p in directory.iterdir()}
        if not expected | {"native_identity.json"} <= actual or actual - expected - optional - {"native_identity.json"}:
            raise ValueError("Mol-1 duplicate/unexpected sample output or missing native identity")
        identity = json.loads((directory / "native_identity.json").read_text())
        if identity["job"] != job["name"] or identity["seed"] != seed or identity["samples"] != count:
            raise ValueError("Mol-1 native output identity differs from request")
    if (root / "ERR").exists() and any((root / "ERR").iterdir()):
        raise RuntimeError("Mol-1 reported native inference errors")
    return samples
