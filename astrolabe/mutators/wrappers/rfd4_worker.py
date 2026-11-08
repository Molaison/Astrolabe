"""Private RFD4 boundary. CPU preparation never constructs a Proteina model."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig

_AA3 = dict(zip("ACDEFGHIKLMNPQRSTVWY", "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()))
_AA1 = {value: key for key, value in _AA3.items()}
_SIDECHAINS = dict(zip(_AA3, (
    "CB", "CB SG", "CB CG OD1 OD2", "CB CG CD OE1 OE2", "CB CG CD1 CD2 CE1 CE2 CZ", "",
    "CB CG ND1 CD2 CE1 NE2", "CB CG1 CG2 CD1", "CB CG CD CE NZ", "CB CG CD1 CD2",
    "CB CG SD CE", "CB CG OD1 ND2", "CB CG CD", "CB CG CD OE1 NE2",
    "CB CG CD NE CZ NH1 NH2", "CB OG", "CB OG1 CG2", "CB CG1 CG2",
    "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2", "CB CG CD1 CD2 CE1 CE2 CZ OH")))


def _compose(settings, *, inputs="<ANNOTATED_INPUT>", out_dir="<REQUEST_OUTPUT>", seed=0):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(settings["source_dir"]) / "configs")):
        cfg = compose(config_name="inference", overrides=[f"recycle_strategy={settings['generation']['recycle_strategy']}"])
    with open_dict(cfg):
        for key, value in {
            "inputs": inputs, "out_dir": out_dir, "seed": seed,
            "hf_ckpt_path": settings["model_dir"], "hf_ckpt_revision": None,
            "ckpt_path": None, "ckpt_name": None, "autoencoder_ckpt_path": None,
            "adapter_path": None, "inference_backend": "native",
            "metrics": None, "metrics_tags": [], "delete_failing_pre_oracle": False,
            "search": {"checkpoints": None, "save_states": False, "resume_from": None, "resume_seed": None},
            "skip_existing": False, "num_replicates": 1, "num_workers": 0,
            "batch_size": 1, "gen_njobs": 1, "eval_njobs": 1, "job_id": None,
            "compile_lengths": None, "compress_outputs": False, "save_chain_of_interest": False,
            "precision": settings["generation"]["precision"], "prevalidate_inputs": True,
        }.items():
            cfg[key] = value
        for key in ("nsteps", "partial_t_backbone", "partial_t_latent", "partial_trajectory_fraction_steps"):
            cfg.generation.args[key] = settings["generation"].get(key)
        cfg.generation.args.couple_partial_t = False
    OmegaConf.resolve(cfg)
    bands = cfg.generation.n_recycle_schedule
    if bands is not None and sum(int(b[0]) for b in bands) != cfg.generation.args.nsteps:
        raise ValueError("Native recycle schedule does not cover the requested nsteps")
    return cfg


def environment_identity(settings):
    """Metadata, actual dependency files, and resolved Hydra config without Torch import."""
    from omegaconf import OmegaConf
    distributions = []
    files = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name", "")
        distributions.append([name, dist.version])
        for entry in dist.files or ():
            path = Path(dist.locate_file(entry)).absolute()
            if (path.suffix in {".py", ".pth"} or ".so" in path.name
                    or path.name in {"METADATA", "RECORD", "direct_url.json"}):
                files.add(str(path))
    lib = Path(sysconfig.get_config_var("LIBDIR") or "") / (sysconfig.get_config_var("LDLIBRARY") or "")
    if lib.is_file():
        files.add(str(lib))
    files.add(str(Path(sys.executable).absolute()))
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "libc": list(platform.libc_ver()), "distributions": sorted(distributions),
            "dependency_files": sorted(files), "sys_path": sys.path,
            "effective_config": OmegaConf.to_container(_compose(settings), resolve=True)}


def _raw_structure(path, *, ligand_codes=()):
    import numpy as np
    from biotite.structure.io import pdb, pdbx
    from atomworks.io.utils.atom_array_plus import as_atom_array_plus
    path = Path(path)
    if path.suffix.lower() == ".pdb":
        file = pdb.PDBFile.read(path)
        if file.get_model_count() != 1:
            raise ValueError("Exactly one input structural model is required")
        arr = file.get_structure(model=1, altloc="all", extra_fields=["charge", "b_factor", "occupancy"], include_bonds=True)
    else:
        file = pdbx.CIFFile.read(path)
        if pdbx.get_model_count(file) != 1:
            raise ValueError("Exactly one structural model is required")
        fields = [name for name in ("auth_asym_id", "label_asym_id", "auth_seq_id", "label_seq_id",
                                    "auth_atom_id", "label_atom_id", "label_alt_id") if name in file.block["atom_site"]]
        arr = pdbx.get_structure(file, model=1, altloc="all", use_author_fields=True,
                                 extra_fields=fields + ["charge", "b_factor", "occupancy"], include_bonds=False)
    if len(arr) == 0 or not np.isfinite(arr.coord).all():
        raise ValueError("Structure must be nonempty with finite coordinates for every atom")
    if "label_alt_id" in arr.get_annotation_categories():
        if any(str(x) not in {"", ".", "?"} for x in arr.label_alt_id):
            raise ValueError("Alternate conformations require an explicit upstream choice")
    if "altloc_id" in arr.get_annotation_categories() and any(str(x).strip() not in {"", ".", "?"} for x in arr.altloc_id):
        raise ValueError("Alternate conformations require an explicit upstream choice")
    if path.suffix.lower() != ".pdb":
        arr = pdbx.get_structure(file, model=1, altloc="first", use_author_fields=True,
                                 extra_fields=fields + ["charge", "b_factor", "occupancy"], include_bonds=True)
    arr = as_atom_array_plus(arr)
    if ligand_codes:
        from atomworks.io.utils.ccd import (atom_array_from_ccd_code,
                                          build_ccd_entries_from_cif_block, get_chem_comp_type)
        entries = (build_ccd_entries_from_cif_block(file.block, on_mismatch="ignore")
                   if path.suffix.lower() != ".pdb" else {})
        templates = {}
        for code in set(ligand_codes):
            template = entries.get(code)
            if template is None:
                template = atom_array_from_ccd_code(code)
            names = set(map(str, template.atom_name))
            # Do not relabel a polymer CCD (or a custom polymer backbone) as a ligand.
            kind = str(template.chem_comp_type[0]).upper()
            bundled_kind = get_chem_comp_type(code, atom_names=names).upper()
            if (kind != "NON-POLYMER" or bundled_kind != "NON-POLYMER"
                    or {"N", "CA", "C"} <= names or {"O3'", "P"} <= names):
                raise ValueError(f"Ligand CCD must be a non-polymer, not RNA/protein: {code}")
            templates[code] = template
            selected = np.flatnonzero(arr.res_name == code)
            by_name = {str(atom.atom_name): atom for atom in template}
            for index in selected:
                name = str(arr.atom_name[index])
                if name not in by_name:
                    raise ValueError(f"Ligand atom is not in retained CCD: {code}/{name}")
                atom = by_name[name]
                if str(arr.element[index]).upper() != str(atom.element).upper():
                    raise ValueError(f"Ligand element disagrees with retained CCD: {code}/{name}")
                # atom_site often omits formal charge; the authored CCD is authoritative then.
                explicit = path.suffix.lower() == ".pdb"
                if not explicit and "pdbx_formal_charge" in file.block["atom_site"]:
                    values = file.block["atom_site"]["pdbx_formal_charge"].as_array(str)
                    explicit = str(values[index]) not in {"", ".", "?"}
                if explicit and int(arr.charge[index]) != int(atom.charge):
                    raise ValueError(f"Ligand charge disagrees with retained CCD: {code}/{name}")
                arr.charge[index] = atom.charge
        arr._custom_ccd_registry = templates
        arr.set_annotation("chem_comp_type", np.array([
            (str(templates[str(name)].chem_comp_type[0]) if str(name) in templates
             else get_chem_comp_type(str(name))) for name in arr.res_name], dtype=str))
    return arr


def _ligand_chemistry(residue):
    """Atom-name keyed heavy chemistry, independent of atom order and rigid motion."""
    import numpy as np
    heavy = residue[~np.isin(residue.element, ["H", "D"])]
    atoms = sorted((str(atom.atom_name), str(atom.element).upper(), int(atom.charge))
                   for atom in heavy)
    bonds = []
    if heavy.bonds is not None:
        for a, b, order in heavy.bonds.as_array():
            if int(order) == 0:
                raise ValueError("Ligand bond order is unspecified; explicit CCD chemistry is required")
            first, second = sorted((str(heavy.atom_name[a]), str(heavy.atom_name[b])))
            bonds.append((first, second, int(order)))
    if not atoms or len({atom[0] for atom in atoms}) != len(atoms):
        raise ValueError("Ligand heavy atoms must be nonempty and uniquely named")
    if len(atoms) > 1 and not bonds:
        raise ValueError("Ligand intramolecular topology is missing")
    return {"heavy_atoms": [list(atom) for atom in atoms],
            "heavy_bonds": [list(bond) for bond in sorted(bonds)]}


def _inventory(arr, chain_specs=None):
    """Validate canonical protein and explicitly declared ligand heavy chemistry."""
    import numpy as np
    import biotite.structure as struc
    residues = []
    seen = set()
    for start, stop in zip(struc.get_residue_starts(arr), struc.get_residue_starts(arr, add_exclusive_stop=True)[1:]):
        residue = arr[start:stop]
        name = str(residue.res_name[0])
        if len(set(residue.res_name)) != 1:
            raise ValueError(f"Noncanonical or mixed residue chemistry: {name}")
        key = (str(residue.chain_id[0]), int(residue.res_id[0]), str(residue.ins_code[0]))
        if key in seen:
            raise ValueError(f"Discontiguous or duplicate residue identity: {key}")
        seen.add(key)
        names = list(map(str, residue.atom_name))
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate atoms or alternate conformers: {key}")
        spec = chain_specs.get(key[0]) if chain_specs is not None else None
        if chain_specs is not None and spec is None:
            raise ValueError(f"Undeclared physical chain: {key[0]}")
        if spec is not None and spec.get("kind", "protein") == "ligand":
            if name != spec["sequence"]:
                raise ValueError(f"Ligand chain has mixed or changed CCD identity: {key[0]}/{name}")
            template = arr._custom_ccd_registry[name]
            if _ligand_chemistry(residue) != _ligand_chemistry(template):
                raise ValueError(f"Ligand heavy atoms, charges or topology disagree with retained CCD: {key}")
            residues.append((int(start), int(stop)))
            continue
        if name not in _AA1:
            raise ValueError(f"Noncanonical protein residue chemistry: {name}")
        heavy = residue[~np.isin(residue.element, ["H", "D"])]
        required = set(("N CA C O " + _SIDECHAINS[_AA1[name]]).split())
        actual = set(map(str, heavy.atom_name))
        if actual - {"OXT"} != required:
            raise ValueError(f"Incomplete/noncanonical heavy atoms at {key}: expected {sorted(required)}, got {sorted(actual)}")
        for atom in heavy:
            expected = "S" if atom.atom_name in {"SG", "SD"} else str(atom.atom_name)[0]
            if atom.element != expected:
                raise ValueError(f"Atom element disagrees with canonical chemistry at {key}/{atom.atom_name}")
        residues.append((int(start), int(stop)))
    chains = list(dict.fromkeys(map(str, arr.chain_id)))
    for chain in chains:
        blocks = [x for x in residues if str(arr.chain_id[x[0]]) == chain]
        if chain_specs is not None and chain_specs[chain].get("kind", "protein") == "ligand":
            if len(blocks) != 1:
                raise ValueError("A ligand chain must contain exactly one CCD residue")
            continue
        if any("OXT" in arr.atom_name[a:b] for a, b in blocks[:-1]):
            raise ValueError("OXT is supported only on a physical chain's last residue")
    if chain_specs is not None and arr.bonds is not None:
        for a, b, _ in arr.bonds.as_array():
            first, second = str(arr.chain_id[a]), str(arr.chain_id[b])
            if first != second and any(chain_specs[c].get("kind", "protein") == "ligand"
                                       for c in (first, second)):
                raise ValueError("Covalent protein-ligand or inter-ligand bonds are unsupported")
    return residues


def _normalize(root, request):
    import numpy as np
    from atomworks.io.transforms.atom_array import add_id_and_entity_annotations, add_iid_annotations
    from atomworks.io.utils.annotator import ensure_annotations
    specs = {chain["chain"]: chain for chain in request["chains"]}
    ligand_codes = {chain["sequence"] for chain in request["chains"]
                    if chain.get("kind", "protein") == "ligand"}
    arr = _raw_structure(Path(root) / request["source_file"], ligand_codes=ligand_codes)
    original_author_chain = arr.chain_id.copy()
    if "label_asym_id" in arr.get_annotation_categories():
        arr.set_annotation("chain_id", arr.label_asym_id.copy())
    fields = set(arr.get_annotation_categories())
    expected = {chain["chain"] for chain in request["chains"]}
    # A label chain is the physical chain. Author names are accepted only as an unambiguous alias.
    physical = arr.label_asym_id if "label_asym_id" in fields else arr.chain_id
    physical_ids = list(dict.fromkeys(map(str, physical)))
    author_ids = list(dict.fromkeys(map(str, original_author_chain)))
    if expected == set(physical_ids):
        requested = np.array(physical, dtype=str)
    elif expected == set(author_ids) and len(author_ids) == len(physical_ids):
        requested = np.array(original_author_chain, dtype=str)
    else:
        raise ValueError(f"Requested physical chains do not bijectively cover input chains: {expected}")
    # The alias must be bijective per atom, not merely equal in number of unique IDs.
    if any(len(set(requested[physical == label])) != 1 for label in physical_ids):
        raise ValueError("Author chain names do not bijectively identify physical chains")
    arr.set_annotation("chain_id", requested.copy())
    blocks = _inventory(arr, specs)
    chain_index = {chain["chain"]: i for i, chain in enumerate(request["chains"])}
    normalized = {chain: (chr(65 + i) if i < 26 else f"X{i+1}") for chain, i in chain_index.items()}
    positions = {chain: 0 for chain in expected}
    sequence = {chain: [] for chain in expected}
    mapping = []
    residue_ids = np.empty(len(arr), dtype=int)
    labels = np.empty(len(arr), dtype=object)
    native_chains = np.empty(len(arr), dtype=object)
    for start, stop in blocks:
        chain = str(requested[start])
        ligand = specs[chain].get("kind", "protein") == "ligand"
        positions[chain] += 1
        ordinal = positions[chain]
        if not np.all(requested[start:stop] == chain):
            raise ValueError("One parsed residue spans multiple physical chain labels")
        label = str(physical[start])
        auth_residue = str(arr.auth_seq_id[start]) if "auth_seq_id" in fields else str(arr.res_id[start])
        label_residue = str(arr.label_seq_id[start]) if "label_seq_id" in fields else None
        mapping.append({"chain": chain, "position": ordinal, "native_chain": normalized[chain],
                        "native_residue": ordinal, "author_chain": str(original_author_chain[start]),
                        "author_residue": auth_residue, "insertion_code": str(arr.ins_code[start]),
                        "label_chain": label, "label_residue": label_residue,
                        "residue_name": str(arr.res_name[start]),
                        "atoms": [{"name": str(arr.atom_name[j]), "element": str(arr.element[j]),
                                   "charge": int(arr.charge[j]) if "charge" in fields else None,
                                   "source_coordinates_angstrom": arr.coord[j].astype(float).tolist(),
                                   "native_present": bool(ligand or arr.atom_name[j] != "OXT"),
                                   "omission_reason": "native_RFD4_does_not_represent_terminal_OXT" if not ligand and arr.atom_name[j] == "OXT" else None,
                                   "author_atom": str(arr.auth_atom_id[j]) if "auth_atom_id" in fields else str(arr.atom_name[j]),
                                   "label_atom": str(arr.label_atom_id[j]) if "label_atom_id" in fields else str(arr.atom_name[j])}
                                  for j in range(start, stop)]})
        if ligand:
            mapping[-1].update(kind="ligand", ccd=str(arr.res_name[start]),
                               retained_chemistry=_ligand_chemistry(arr[start:stop]))
        residue_ids[start:stop] = ordinal
        labels[start:stop] = chain
        native_chains[start:stop] = normalized[chain]
        sequence[chain].append(str(arr.res_name[start]) if ligand else _AA1[str(arr.res_name[start])])
    for chain in request["chains"]:
        if "".join(sequence[chain["chain"]]) != chain["sequence"]:
            raise ValueError(f"Input sequence/length mismatch for physical chain {chain['chain']}")
    # Raw readers do not deserialize any inherited model conditions.
    for name in list(arr.get_annotation_categories()):
        if name.startswith(("C_", "condition_", "mask_")) or name in {
                "chain_iid", "molecule_iid", "pn_unit_iid", "chain_entity", "molecule_entity", "pn_unit_entity",
                "label_asym_id", "auth_asym_id", "label_seq_id", "auth_seq_id", "auth_atom_id", "label_atom_id"}:
            arr.del_annotation(name)
    arr.set_annotation("chain_id", np.array(native_chains, dtype=str))
    arr.set_annotation("res_id", residue_ids)
    arr.set_annotation("ins_code", np.full(len(arr), "", dtype=str))
    arr.set_annotation("physical_chain", np.array(labels, dtype=str))
    arr.set_annotation("transformation_id", np.full(len(arr), "1", dtype=str))
    omitted = {(entry["chain"], entry["position"], atom["name"])
               for entry in mapping for atom in entry["atoms"] if not atom["native_present"]}
    for condition in request["plan"]["hbonds"]:
        atom = condition["atom"]
        if (atom["chain"], atom["position"], atom["atom"]) in omitted:
            raise ValueError(f"C_HB cannot condition an omitted terminal OXT atom: {atom}")
    omit_mask = np.array([(str(chain), int(position), str(name)) in omitted
                          for chain, position, name in zip(arr.physical_chain, arr.res_id, arr.atom_name)])
    arr = arr[~omit_mask]
    # Chain classification and serialization must use this input's CCD context.
    from atomworks.io.utils.ccd import custom_ccd_residues
    with custom_ccd_residues(arr._custom_ccd_registry):
        ensure_annotations(arr, "is_polymer")
    arr = add_id_and_entity_annotations(arr)
    arr = add_iid_annotations(arr)
    return arr, mapping


def _projection(mapping, native_atom_count):
    omitted = []
    for entry in mapping:
        for atom in entry["atoms"]:
            if not atom["native_present"]:
                omitted.append({
                    "atom_ref": {"chain": entry["chain"], "position": entry["position"], "atom": atom["name"]},
                    "source_coordinates_angstrom": atom["source_coordinates_angstrom"],
                    "author_chain": entry["author_chain"], "author_residue": entry["author_residue"],
                    "insertion_code": entry["insertion_code"], "author_atom": atom["author_atom"],
                    "label_chain": entry["label_chain"], "label_residue": entry["label_residue"],
                    "label_atom": atom["label_atom"], "reason": atom["omission_reason"],
                })
    return {"policy": "omit_only_validated_physical_chain_terminal_OXT",
            "source_atom_count": sum(len(entry["atoms"]) for entry in mapping),
            "native_input_atom_count": native_atom_count, "omitted_atoms": omitted}


def _condition(arr, request):
    import numpy as np
    from atomworks.ml.conditions import C_SEQ, C_CRD, C_IDX, CONDITIONS
    from rfproteina.conditions import C_HB, HBondEnum, SEQUENCE_PROBABILITY_CONDITION_BY_TOKEN
    for condition in CONDITIONS:
        for name in (condition.full_name, *condition.aliases):
            if name in arr.get_annotation_categories():
                arr.del_annotation(name)
    released = {(x["chain"], x["position"]) for x in request["plan"]["released"]}
    fixed = {(x["chain"], x["position"]) for x in request["plan"]["fixed_coordinates"]}
    ligands = {(chain["chain"], 1) for chain in request["chains"]
               if chain.get("kind", "protein") == "ligand"}
    if released & ligands:
        raise ValueError("Ligand residues must never be released")
    fixed |= ligands
    sites = list(zip(map(str, arr.physical_chain), map(int, arr.res_id)))
    release_mask = np.array([site in released for site in sites])
    fixed_mask = np.array([site in fixed for site in sites]) & ~np.isin(arr.element, ["H", "D"])
    fixed_mask |= np.array([site in ligands for site in sites])
    C_SEQ.set_annotation_from_ground_truth(arr, ~release_mask)
    C_CRD.set_annotation_from_ground_truth(arr, fixed_mask)
    C_IDX.set_annotation(arr, np.ones(len(arr), dtype=bool))
    hbond = C_HB.default_annotation(arr)
    seen = set()
    for condition in request["plan"]["hbonds"]:
        atom = condition["atom"]
        key = (atom["chain"], atom["position"], atom["atom"])
        if key in seen:
            raise ValueError(f"Duplicate C_HB atom condition: {key}")
        seen.add(key)
        selected = np.array([site == key[:2] for site in sites]) & (arr.atom_name == key[2])
        if selected.sum() != 1 or not fixed_mask[selected].all():
            raise ValueError(f"C_HB atom is missing or not a stable heavy atom: {key}")
        hbond[selected] = HBondEnum[condition["role"]].value
    C_HB.set_annotation(arr, hbond)
    excluded = set(request["plan"]["excluded_amino_acids"])
    # An unrestricted proposal must leave probability conditioning unspecified.
    # A dense 1/20 vector is an explicit soft distribution, not a neutral mask.
    # Canonical output identity is still checked at the validation boundary.
    if not excluded:
        return arr
    # Preserve the existing explicit-exclusion protocol when exclusions are requested.
    allowed = set(_AA3) - excluded
    for token, condition in SEQUENCE_PROBABILITY_CONDITION_BY_TOKEN.items():
        values = condition.default_annotation(arr)
        probability = 1.0 / len(allowed) if token in _AA1 and _AA1[token] in allowed else 0.0
        values[release_mask] = probability
        condition.set_annotation(arr, values)
    return arr


def _annotated(path):
    from atomworks.io.config import ParseConfig
    from atomworks.io.parser import parse
    config = ParseConfig(model=1, build_assembly=None, add_missing_atoms=False,
                         remove_ccds=(), remove_waters=False, fix_ligands_at_symmetry_centers=False,
                         fix_arginines=False, long_bond_policy="keep", struct_conn_distance_policy="keep",
                         hydrogen_policy="keep", load_standard_annotations=True,
                         return_atom_array_plus=True, add_bond_types_from_struct_conn=("covale", "disulf"))
    return parse(path, config=config)["asym_unit"][0]


def _check_annotations(expected, actual):
    import numpy as np
    from atomworks.ml.conditions import CONDITIONS
    if len(actual) != len(expected):
        raise ValueError("Annotated input round-trip changed atom inventory")
    for field in ("atom_name", "element", "res_name", "res_id", "chain_id", "charge", "chem_comp_type"):
        if field in expected.get_annotation_categories() and not np.array_equal(getattr(expected, field), getattr(actual, field)):
            raise ValueError(f"Annotated CIF changed input {field}")
    def bond_inventory(array):
        if array.bonds is None:
            return set()
        return {(min(int(a), int(b)), max(int(a), int(b)), int(order))
                for a, b, order in array.bonds.as_array()}
    if bond_inventory(expected) != bond_inventory(actual):
        raise ValueError("Annotated CIF changed covalent topology; this chain mapping is unsupported")
    if not np.allclose(expected.coord, actual.coord, atol=0.002, rtol=0):
        raise ValueError("Annotated CIF changed input coordinates")
    for condition in CONDITIONS:
        if condition.has_annotation(expected):
            if not condition.has_annotation(actual):
                raise ValueError(f"Annotation lost in CIF: {condition.name}")
            first, second = condition.annotation(expected), condition.annotation(actual)
            if np.asarray(first).dtype.kind in "fc":
                equal = np.allclose(first, second, equal_nan=True)
            else:
                equal = np.array_equal(first, second)
            if not equal:
                raise ValueError(f"Annotation changed in CIF: {condition.name}")
        elif condition.has_annotation(actual):
            raise ValueError(f"Unexpected inherited condition in CIF: {condition.name}")


def prepare(payload):
    if payload["request"]["plan"].get("reconstruction") is not None:
        return _unindexed_worker().prepare(payload, sys.modules[__name__])
    from atomworks.io.utils.io_utils import to_cif_file
    root, request = Path(payload["root"]), payload["request"]
    arr, mapping = _normalize(root, request)
    arr = _condition(arr, request)
    path = root / "annotated_input.cif"
    to_cif_file(arr, path, save_standard_annotations=True, chain_disambiguation="chain_iid")
    _check_annotations(arr, _annotated(path))
    return {"mapping": mapping, "atom_count": len(arr), "projection": _projection(mapping, len(arr))}


def validate(payload):
    if payload["request"]["plan"].get("reconstruction") is not None:
        return _unindexed_worker().validate(payload, sys.modules[__name__])
    import numpy as np
    root, request = Path(payload["root"]), payload["request"]
    arr, mapping = _normalize(root, request)
    _check_annotations(_condition(arr, request), _annotated(root / "annotated_input.cif"))
    ligand_codes = {chain["sequence"] for chain in request["chains"]
                    if chain.get("kind", "protein") == "ligand"}
    output = _raw_structure(root / "generated.cif", ligand_codes=ligand_codes)
    by_native = {entry["native_chain"]: entry["chain"] for entry in mapping}
    if set(map(str, output.chain_id)) != set(by_native):
        raise ValueError("Native output changed physical chain inventory")
    specs = {chain["chain"]: chain for chain in request["chains"]}
    blocks = _inventory(output, {native: specs[chain] for native, chain in by_native.items()})
    source_ligands = {entry["chain"]: entry["retained_chemistry"] for entry in mapping
                      if entry.get("kind") == "ligand"}
    sequences = {c["chain"]: [] for c in request["chains"]}
    native_counts = {chain: 0 for chain in by_native}
    atoms = {}
    output_mapping = []
    for start, stop in blocks:
        native = str(output.chain_id[start])
        native_counts[native] += 1
        chain, ordinal = by_native[native], native_counts[native]
        ligand = specs[chain].get("kind", "protein") == "ligand"
        if ligand and _ligand_chemistry(output[start:stop]) != source_ligands[chain]:
            raise ValueError(f"Native output changed ligand heavy atoms, elements, charges or bond topology: {chain}")
        sequences[chain].append(str(output.res_name[start]) if ligand else _AA1[str(output.res_name[start])])
        output_mapping.append({"chain": chain, "position": ordinal,
                               "output_chain": native, "output_residue": int(output.res_id[start]),
                               "output_insertion_code": str(output.ins_code[start])})
        for index in range(start, stop):
            atoms[(chain, ordinal, str(output.atom_name[index]))] = index
    released = {(x["chain"], x["position"]) for x in request["plan"]["released"]}
    fixed = {(x["chain"], x["position"]) for x in request["plan"]["fixed_coordinates"]}
    fixed |= {(chain["chain"], 1) for chain in request["chains"]
              if chain.get("kind", "protein") == "ligand"}
    excluded = set(request["plan"]["excluded_amino_acids"])
    for chain in request["chains"]:
        actual = "".join(sequences[chain["chain"]])
        if chain.get("kind", "protein") == "ligand":
            if len(sequences[chain["chain"]]) != 1 or actual != chain["sequence"]:
                raise ValueError(f"Native output changed ligand CCD identity: {chain['chain']}")
            continue
        if len(actual) != len(chain["sequence"]):
            raise ValueError(f"Native output changed chain length: {chain['chain']}")
        for ordinal, (old, new) in enumerate(zip(chain["sequence"], actual), 1):
            if (chain["chain"], ordinal) not in released and old != new:
                raise ValueError(f"Native output changed a fixed identity: {chain['chain']}/{ordinal}")
            if (chain["chain"], ordinal) in released and new in excluded:
                raise ValueError(f"Native output violated a sequence exclusion: {chain['chain']}/{ordinal}")
    source_coords, generated_coords = [], []
    for index in range(len(arr)):
        site = (str(arr.physical_chain[index]), int(arr.res_id[index]))
        if site in fixed and arr.element[index] not in {"H", "D"}:
            key = (*site, str(arr.atom_name[index]))
            if key not in atoms:
                raise ValueError(f"Fixed heavy atom missing from output: {key}")
            source_coords.append(arr.coord[index])
            generated_coords.append(output.coord[atoms[key]])
    rmsd = unaligned = None
    rotation = translation = None
    if source_coords:
        first, second = np.asarray(source_coords, dtype=np.float64), np.asarray(generated_coords, dtype=np.float64)
        unaligned = float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))
        x, y = first - first.mean(0), second - second.mean(0)
        u, _, vt = np.linalg.svd(y.T @ x)
        correction = np.eye(3)
        correction[2, 2] = np.linalg.det(u @ vt)
        transform = u @ correction @ vt
        shift = first.mean(0) - second.mean(0) @ transform
        rmsd = float(np.sqrt(np.mean(np.sum((second @ transform + shift - first) ** 2, axis=1))))
        rotation, translation = transform.tolist(), shift.tolist()
        if not np.isfinite(rmsd) or rmsd > payload["max_fixed_atom_rmsd"]:
            raise ValueError(f"Native-representable fixed heavy atom RMSD {rmsd:.6g} A exceeds engineering gate {payload['max_fixed_atom_rmsd']} A")
    return {"sequences": [[c["chain"], "".join(sequences[c["chain"]])] for c in request["chains"]],
            "mapping": mapping, "output_mapping": output_mapping,
            "projection": _projection(mapping, len(arr)),
            "input_atom_count": len(arr), "generated_atom_count": len(output),
            "fixed_atom_rmsd_scope": "native_representable_fixed_heavy_atoms_excluding_projected_OXT",
            "fixed_heavy_atom_count": len(source_coords), "fixed_atom_rmsd_angstrom": rmsd,
            "fixed_atom_unaligned_rmsd_angstrom": unaligned,
            "fixed_atom_alignment": {"rotation": rotation, "translation": translation},
            "max_fixed_atom_rmsd_angstrom": payload["max_fixed_atom_rmsd"],
            "fixed_atom_gate": "engineering_tolerance_not_biologically_validated",
            "structure_status": "generated_unrefolded"}


def _unindexed_worker():
    if __package__:
        from . import rfd4_unindexed_worker
    else:
        import rfd4_unindexed_worker
    return rfd4_unindexed_worker


def interaction_metrics(payload):
    """The native terminal interaction reward inputs, without model construction."""
    import math
    from rfproteina.metrics.interaction_metrics import get_interaction_metrics
    values = get_interaction_metrics(_annotated(Path(payload["structure_path"])))
    required = {"hbond_atom_pair_count_coi", "hydrophobic_atom_pair_count_coi"}
    for key in required:
        if key not in values or not math.isfinite(float(values[key])):
            raise ValueError(f"Native interaction metric is missing or nonfinite: {key}")
    return {key: float(value) if math.isfinite(float(value)) else None
            for key, value in values.items()}


def _load(settings):
    from omegaconf import OmegaConf
    from rfproteina.proteina import Proteina
    from rfproteina.generate import run_generation
    model = Proteina.from_pretrained(settings["model_dir"], local_files_only=True)

    def infer(payload):
        root = Path(payload["root"])
        # A unique parent also isolates native results_<run>_<job>.csv skip checks.
        run_root = root / "native"
        run_root.mkdir(exist_ok=False)
        cfg = _compose(settings, inputs=str(root / "annotated_input.cif"),
                       out_dir=str(run_root / "outputs"), seed=payload["request"]["seed"])
        resolved = OmegaConf.to_container(cfg, resolve=True)
        (root / "native_config.json").write_text(json.dumps(resolved, sort_keys=True, indent=2))
        run_generation(cfg, model=model)
        outputs = sorted((run_root / "outputs/inference_samples").rglob("*_generated.cif"))
        if len(outputs) != 1:
            raise RuntimeError(f"Native RFD4 must emit exactly one full generated CIF, got {len(outputs)}")
        return {"structure_path": str(outputs[0]), "provenance": {"effective_config": resolved,
                "loader": "Proteina.from_pretrained(local_files_only=True)",
                "generation": "run_generation(cfg, model=resident)", "uplift": False,
                "structure_status": "generated_unrefolded"}}

    return infer, {"backend": "native_rfd4", "model_dir": settings["model_dir"]}


def create_handler(settings):
    from astrolabe.model_execution import kernel_handler
    return kernel_handler(settings, _load, torch_device="cuda")


if __name__ == "__main__":
    payload = json.load(sys.stdin)
    operation = sys.argv[1]
    result = {"identity": environment_identity, "prepare": prepare, "validate": validate,
              "interaction_metrics": interaction_metrics}[operation](payload)
    print("RFD4_RESULT=" + json.dumps(result, sort_keys=True, allow_nan=False))
