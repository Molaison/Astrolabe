"""Native variable-length unindexed scaffolding; no model or inference engine here.

``build_reconstruction_template(source, request)`` and ``check_template()`` are
CPU authoring helpers. Length bounds describe the ENTIRE generated protein,
including placed motifs. Original context residues disappear in native inference; they are
not extra output residues. Source tags are native chain+res_id labels, not user
lineage IDs (SA_SRC/SA_CRDSRC have only ten characters of storage).
"""
from __future__ import annotations

from pathlib import Path


def build_reconstruction_template(source, request):
    """Return a collapsed native template and its request-linked motif manifest.

    ``source`` is the complete normalized parent from rfd4_worker._normalize().
    It is never mutated. Each motif is a complete native heavy-atom residue in
    its own index group; no original protein scaffold is retained. Context and
    scaffold MUST share chain_id: the native insertion head assigns within chains.
    """
    import numpy as np
    from atomworks.enums import ChainType
    from atomworks.io.transforms.atom_array import add_id_and_entity_annotations, add_iid_annotations
    from atomworks.io.utils.annotator import ensure_annotations
    from atomworks.io.utils.atom_array_plus import concatenate_atom_array_plus
    from atomworks.io.utils.standard_annotations import S_SEGMIN
    from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ, CONDITIONS
    from atomworks.ml.utils.expandable_segment import insert_expandable_segment
    from rfproteina.conditions import C_HB, C_IDXGRP, HBondEnum, SEQUENCE_PROBABILITY_CONDITION_BY_TOKEN
    from rfproteina.transforms.inference_annotations import build_src_component_annotation
    from rfproteina.utils.standard_annotations import SA_SRC

    plan = request["plan"]
    reconstruction = plan["reconstruction"]
    proteins = [c for c in request["chains"] if c.get("kind", "protein") == "protein"]
    ligands = [c for c in request["chains"] if c.get("kind", "protein") == "ligand"]
    if len(proteins) != 1 or not ligands or len(proteins) + len(ligands) != len(request["chains"]):
        raise ValueError("Reconstruction requires one protein binder and fixed single-CCD ligand chains")
    if reconstruction["chain"] != proteins[0]["chain"]:
        raise ValueError("Reconstruction chain must be the sole protein binder")
    lower, upper = reconstruction["length_min"], reconstruction["length_max"]
    if (type(lower) is not int or type(upper) is not int
            or not 1 <= lower <= upper or len(reconstruction["motifs"]) > lower):
        raise ValueError("Reconstruction needs positive inclusive length bounds accommodating every motif")
    native_binder = str(source.chain_id[source.physical_chain == reconstruction["chain"]][0])
    ligand_chains = {c["chain"] for c in ligands}
    ligand_mask = np.isin(source.physical_chain, list(ligand_chains))
    heavy = ~np.isin(source.element, ["H", "D"])
    chunks = [source[ligand_mask & heavy].copy()]
    manifest = []
    used_sites, used_ids = set(), set()
    for ordinal, motif in enumerate(reconstruction["motifs"], 1):
        site = (motif["source"]["chain"], motif["source"]["position"])
        identifier, names = motif["id"], list(motif["atoms"])
        if (not isinstance(identifier, str) or not identifier or identifier in used_ids
                or site in used_sites or site[0] != reconstruction["chain"]
                or not names or len(set(names)) != len(names)):
            raise ValueError("Motifs need unique lineage IDs, distinct binder residues and nonempty unique atoms")
        selected = (source.physical_chain == site[0]) & (source.res_id == site[1])
        context = source[selected & heavy].copy()
        if not len(context) or len(set(map(str, context.res_name))) != 1:
            raise ValueError(f"Motif source residue is missing or ambiguous: {site}")
        if any(np.count_nonzero(context.atom_name == name) != 1 for name in names):
            raise ValueError(f"Requested motif atom is not native-representable: {site}/{names}")
        # AddSrcComponentAnnotation deliberately regenerates SA_SRC before expansion.
        # Gaps avoid accidental peptide bonds between independent contexts on reload.
        # The sentinel has res_id=1; these source offsets are never output positions.
        context_residue = 2 * ordinal + 1
        tag = f"{native_binder}{context_residue}"
        if len(tag) > 10:
            raise ValueError("Motif source label exceeds native provenance storage")
        context.set_annotation("chain_id", np.full(len(context), native_binder))
        context.set_annotation("res_id", np.full(len(context), context_residue, dtype=int))
        chunks.append(context)
        manifest.append({"id": identifier, "source": dict(motif["source"]), "atoms": names,
                         "context_chain": native_binder, "context_residue": context_residue, "source_tag": tag,
                         "residue_name": str(context.res_name[0])})
        used_sites.add(site)
        used_ids.add(identifier)

    template = concatenate_atom_array_plus(chunks)
    # Drop all inherited conditions, including aliases, before authoring the new problem.
    for condition in CONDITIONS:
        for name in (condition.full_name, *condition.aliases):
            if name in template.get_annotation_categories():
                template.del_annotation(name)
    template = add_id_and_entity_annotations(template, overwrite=True)
    template = add_iid_annotations(template, overwrite=True)
    ensure_annotations(template, "chain_type")
    template = insert_expandable_segment(
        template, seg_min=lower, seg_max=upper, new_chain=True,
        new_chain_id=native_binder, new_chain_type=ChainType.POLYPEPTIDE_L)
    # The sentinel is a NEW output chain, but contexts must subsequently be in
    # that same chain for native per-chain insertion assignment. Keep it contiguous.
    scaffold = S_SEGMIN.mask(template)
    context = ~scaffold & (template.chain_id == native_binder)
    template = template[np.r_[np.flatnonzero(scaffold), np.flatnonzero(context),
                              np.flatnonzero(~scaffold & ~context)]]
    template = add_id_and_entity_annotations(template, overwrite=True)
    template = add_iid_annotations(template, overwrite=True)
    scaffold = S_SEGMIN.mask(template)
    physical = np.asarray(template.physical_chain, dtype=object)
    physical[scaffold] = reconstruction["chain"]
    template.set_annotation("physical_chain", physical.astype(str))
    ligand_mask = ~scaffold & np.isin(template.physical_chain, list(ligand_chains))
    fixed = ligand_mask.copy()
    indexed = np.ones(len(template), dtype=bool)
    for motif in manifest:
        context = _context_mask(template, motif)
        indexed[context] = False
        fixed |= context & np.isin(template.atom_name, motif["atoms"])
    C_SEQ.set_annotation_from_ground_truth(template, ~scaffold)
    C_CRD.set_annotation_from_ground_truth(template, fixed)
    C_IDX.set_annotation(template, indexed)
    groups = C_IDXGRP.default_annotation(template)
    # Native UnindexFlaggedTokens shifts the minimum unindexed group to max(indexed)+1.
    # Start there already so the manifest's source groups survive that transform exactly.
    first_group = int(groups[indexed].max()) + 1
    for ordinal, motif in enumerate(manifest):
        motif["source_index_group"] = first_group + ordinal
        groups[_context_mask(template, motif)] = first_group + ordinal
    C_IDXGRP.set_annotation(template, groups)

    hbond = C_HB.default_annotation(template)
    seen = set()
    for condition in plan["hbonds"]:
        ref = condition["atom"]
        key = (ref["chain"], ref["position"], ref["atom"])
        if key in seen:
            raise ValueError(f"Duplicate C_HB atom condition: {key}")
        seen.add(key)
        selected = ligand_mask & (template.physical_chain == key[0]) & (template.atom_name == key[2])
        if key[1] != 1:
            selected[:] = False
        for motif in manifest:
            if (motif["source"]["chain"], motif["source"]["position"]) == key[:2]:
                selected |= _context_mask(template, motif) & (template.atom_name == key[2])
        if selected.sum() != 1 or not fixed[selected].all():
            raise ValueError(f"C_HB atom must be a retained conditioned motif atom or ligand atom: {key}")
        hbond[selected] = HBondEnum[condition["role"]].value
    C_HB.set_annotation(template, hbond)
    excluded = set(plan["excluded_amino_acids"])
    if excluded and not manifest:
        # With no insertions, reuse the unchanged fixed-length exclusion protocol.
        # With motifs, every scaffold position is a potential insertion target:
        # native transfer rejects dense probabilities that conflict with a motif's
        # exact one-hot C_SEQ distribution (even if its identity is not excluded).
        # Do not invent a placement mask or weaken motif identity. Enforce the full
        # requested exclusions on regenerated, non-motif residues after mapping.
        aa1 = dict(zip("ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
                       "ACDEFGHIKLMNPQRSTVWY"))
        allowed = set(aa1.values()) - excluded
        if not allowed:
            raise ValueError("Sequence exclusions leave no canonical scaffold identities")
        for token, condition in SEQUENCE_PROBABILITY_CONDITION_BY_TOKEN.items():
            values = condition.default_annotation(template)
            values[scaffold] = 1.0 / len(allowed) if aa1.get(token) in allowed else 0.0
            condition.set_annotation(template, values)
    SA_SRC.set_annotation(template, build_src_component_annotation(template))
    return template, manifest


def _context_mask(template, motif):
    return ((template.chain_id == motif["context_chain"])
            & (template.res_id == motif["context_residue"]))


def check_template(expected, actual):
    """Check collapsed-template round-trip, including NaN sentinel and native SAs.

    This checks input serialization only. Generated scaffolds are never compared
    to the parent's peptide topology or to the collapsed sentinel's topology.
    """
    import numpy as np
    from atomworks.io.utils.standard_annotations import S_SEGMIN, S_SEGMAX
    from atomworks.ml.conditions import CONDITIONS
    from rfproteina.utils.standard_annotations import SA_SRC
    if len(expected) != len(actual):
        raise ValueError("Reconstruction template round-trip changed atom inventory")
    for field in ("atom_name", "element", "res_name", "res_id", "chain_id", "ins_code", "charge", "chem_comp_type"):
        if field in expected.get_annotation_categories():
            if field not in actual.get_annotation_categories() or not np.array_equal(
                    expected.get_annotation(field), actual.get_annotation(field)):
                raise ValueError(f"Reconstruction template round-trip changed {field}")
    if not np.allclose(expected.coord, actual.coord, atol=0.002, rtol=0, equal_nan=True):
        raise ValueError("Reconstruction template round-trip changed coordinates")
    def bonds(array):
        return set() if array.bonds is None else {
            (min(int(a), int(b)), max(int(a), int(b)), int(order)) for a, b, order in array.bonds.as_array()}
    if bonds(expected) != bonds(actual):
        raise ValueError("Reconstruction template round-trip changed authored bonds")
    for annotation in (*CONDITIONS, S_SEGMIN, S_SEGMAX, SA_SRC):
        if annotation.has_annotation(expected) != annotation.has_annotation(actual):
            raise ValueError(f"Reconstruction template lost or added {annotation.name}")
        if annotation.has_annotation(expected):
            first, second = annotation.annotation(expected), annotation.annotation(actual)
            equal = (np.allclose(first, second, rtol=0, atol=0.002, equal_nan=True)
                     if np.asarray(first).dtype.kind in "fc" else np.array_equal(first, second))
            if not equal:
                raise ValueError(f"Reconstruction template changed {annotation.name}")


def prepare(payload, worker):
    from atomworks.io.utils.io_utils import to_cif_file
    root, request = Path(payload["root"]), payload["request"]
    source, mapping = worker._normalize(root, request)
    template, motifs = build_reconstruction_template(source, request)
    path = root / "annotated_input.cif"
    to_cif_file(template, path, save_standard_annotations=True, chain_disambiguation="chain_iid")
    check_template(template, worker._annotated(path))
    return {"mapping": mapping, "atom_count": len(template), "projection": worker._projection(mapping, len(source)),
            "reconstruction_motifs": motifs,
            "length_ranges": {request["plan"]["reconstruction"]["chain"]: [
                request["plan"]["reconstruction"]["length_min"], request["plan"]["reconstruction"]["length_max"]]},
            "length_semantics": "total_generated_protein_residues_including_motifs",
            "sequence_exclusion_enforcement": ("post_mapping_validation" if motifs
                                                else "native_probability_conditioning_and_validation")}


def _atom_keys(array):
    return [(str(chain), int(position), str(code), str(name))
            for chain, position, code, name in zip(array.chain_id, array.res_id, array.ins_code, array.atom_name)]


def validate(payload, worker):
    """Accept only native, complete, injective provenance and frozen condition values."""
    import numpy as np
    from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
    from rfproteina.conditions import C_HB
    from rfproteina.utils.standard_annotations import SA_UII, SA_CRDSRC, SA_IDXGRPSRC, SA_SRC

    root, request = Path(payload["root"]), payload["request"]
    reconstruction = request["plan"]["reconstruction"]
    source, mapping = worker._normalize(root, request)
    template, motifs = build_reconstruction_template(source, request)
    saved_template = worker._annotated(root / "annotated_input.cif")
    check_template(template, saved_template)
    specs = {chain["chain"]: chain for chain in request["chains"]}
    ligand_codes = {chain["sequence"] for chain in request["chains"] if chain.get("kind", "protein") == "ligand"}
    output = worker._raw_structure(root / "generated.cif", ligand_codes=ligand_codes)
    annotated = worker._annotated(root / "generated.cif")
    # Join by exact serialized atom identities, not array order, geometry or sequence alignment.
    output_keys, annotation_keys = _atom_keys(output), _atom_keys(annotated)
    if (len(set(annotation_keys)) != len(annotation_keys) or len(set(output_keys)) != len(output_keys)
            or set(output_keys) != set(annotation_keys)):
        raise ValueError("Native generated annotations do not bijectively cover the output atoms")
    annotation_index = {key: index for index, key in enumerate(annotation_keys)}
    annotated = annotated[[annotation_index[key] for key in output_keys]]
    for field in ("res_name", "element"):
        if not np.array_equal(output.get_annotation(field), annotated.get_annotation(field)):
            raise ValueError(f"Native generated annotation reader changed {field}")
    if not np.allclose(output.coord, annotated.coord, atol=0.002, rtol=0):
        raise ValueError("Native generated annotation reader changed atom coordinates")
    for annotation in (SA_UII, SA_CRDSRC, SA_IDXGRPSRC, C_IDX, C_SEQ, C_CRD, C_HB):
        if not annotation.has_annotation(annotated):
            raise ValueError(f"Native generated CIF lacks required provenance/condition: {annotation.name}")
    if not C_IDX.mask(annotated).all():
        raise ValueError("Native output still contains unindexed context instead of placed motifs")

    by_native = {entry["native_chain"]: entry["chain"] for entry in mapping}
    if set(map(str, output.chain_id)) != set(by_native):
        raise ValueError("Native reconstruction changed the output chain inventory")
    blocks = worker._inventory(output, {native: specs[chain] for native, chain in by_native.items()})
    source_ligands = {entry["chain"]: entry["retained_chemistry"] for entry in mapping if entry.get("kind") == "ligand"}
    sequences = {chain: [] for chain in specs}
    output_mapping, residues, atoms = [], {}, {}
    for start, stop in blocks:
        native = str(output.chain_id[start])
        chain = by_native[native]
        ordinal = len(sequences[chain]) + 1
        ligand = specs[chain].get("kind", "protein") == "ligand"
        if ligand and worker._ligand_chemistry(output[start:stop]) != source_ligands[chain]:
            raise ValueError(f"Native reconstruction changed fixed ligand chemistry: {chain}")
        sequences[chain].append(str(output.res_name[start]) if ligand else worker._AA1[str(output.res_name[start])])
        residues[(chain, ordinal)] = (start, stop)
        output_mapping.append({"chain": chain, "position": ordinal, "output_chain": native,
                               "output_residue": int(output.res_id[start]), "output_insertion_code": str(output.ins_code[start])})
        for index in range(start, stop):
            atoms[(chain, ordinal, str(output.atom_name[index]))] = index
    length = len(sequences[reconstruction["chain"]])
    if not reconstruction["length_min"] <= length <= reconstruction["length_max"]:
        raise ValueError(f"Native reconstruction length {length} is outside the requested inclusive bounds")
    for chain in specs:
        if specs[chain].get("kind", "protein") == "ligand":
            if len(sequences[chain]) != 1 or sequences[chain][0] != specs[chain]["sequence"]:
                raise ValueError(f"Native reconstruction changed ligand identity/count: {chain}")

    uii = SA_UII.annotation(annotated)
    groups = SA_IDXGRPSRC.annotation(annotated)
    tags = SA_CRDSRC.annotation(annotated)
    expected_uii = np.zeros(len(output), dtype=bool)
    expected_groups = np.full(len(output), -1, dtype=groups.dtype)
    expected_tags = np.full(len(output), ".", dtype=tags.dtype)
    expected_seq = C_SEQ.default_annotation(annotated)
    expected_crd = C_CRD.default_annotation(annotated)
    expected_hb = C_HB.default_annotation(annotated)
    template_crd = C_CRD.annotation(template)
    template_hb = C_HB.annotation(template)
    motif_mapping, placed = [], set()
    source_coords, generated_coords = [], []

    def transfer_conditioned_atoms(source_indices, chain, position):
        for index in source_indices:
            name = str(template.atom_name[index])
            target = atoms.get((chain, position, name))
            if target is None:
                raise ValueError(f"Conditioned atom missing from mapped output: {chain}/{position}/{name}")
            expected_crd[target] = template_crd[index]
            expected_hb[target] = template_hb[index]
            expected_tags[target] = SA_SRC.annotation(template)[index]
            if np.isfinite(template_crd[index]).all():
                source_coords.append(template_crd[index])
                generated_coords.append(output.coord[target])

    for motif in motifs:
        candidates = [(site, bounds) for site, bounds in residues.items()
                      if np.any(groups[bounds[0]:bounds[1]] == motif["source_index_group"])]
        if len(candidates) != 1:
            raise ValueError(f"Missing or noninjective native motif provenance: {motif['id']}")
        (chain, position), (start, stop) = candidates[0]
        if chain != reconstruction["chain"] or (chain, position) in placed:
            raise ValueError(f"Motif mapped outside binder or onto another motif: {motif['id']}")
        if not np.all(output.res_name[start:stop] == motif["residue_name"]):
            raise ValueError(f"Native motif mapping changed conditioned identity: {motif['id']}")
        expected_uii[start:stop] = True
        expected_groups[start:stop] = motif["source_index_group"]
        expected_seq[start:stop] = motif["residue_name"]
        selected = _context_mask(template, motif) & np.isin(template.atom_name, motif["atoms"])
        transfer_conditioned_atoms(np.flatnonzero(selected), chain, position)
        motif_mapping.append({"id": motif["id"], "source": motif["source"],
                              "output": {"chain": chain, "position": position}, "atoms": motif["atoms"]})
        placed.add((chain, position))
    for chain, spec in specs.items():
        if spec.get("kind", "protein") == "ligand":
            start, stop = residues[(chain, 1)]
            expected_seq[start:stop] = spec["sequence"]
            transfer_conditioned_atoms(np.flatnonzero(template.physical_chain == chain), chain, 1)
    for name, actual, expected in (
            ("SA_UII", uii, expected_uii), ("SA_IDXGRPSRC", groups, expected_groups),
            ("SA_CRDSRC", tags, expected_tags), ("C_SEQ", C_SEQ.annotation(annotated), expected_seq),
            ("C_HB", C_HB.annotation(annotated), expected_hb)):
        if not np.array_equal(actual, expected):
            raise ValueError(f"Native mapped condition/provenance mismatch: {name}")
    # CenterOnCoordinateConditionedAtoms shifts C_CRD by the centroid of the
    # conditioned live coordinates (after input-CIF rounding), including contexts
    # exactly once. No best-fit/nearest-atom inference is used for this comparison.
    fixed_mask = C_CRD.mask(saved_template)
    centroid = np.mean(saved_template.coord[fixed_mask], axis=0)
    if not np.allclose(C_CRD.annotation(annotated), expected_crd - centroid, atol=0.002, rtol=0, equal_nan=True):
        raise ValueError("Native mapped conditioned atom coordinates do not correspond to their source atoms")
    excluded = set(request["plan"]["excluded_amino_acids"])
    for position, identity in enumerate(sequences[reconstruction["chain"]], 1):
        if (reconstruction["chain"], position) not in placed and identity in excluded:
            raise ValueError(f"Native reconstruction violated a scaffold sequence exclusion at {position}")

    first, second = np.asarray(source_coords, dtype=np.float64), np.asarray(generated_coords, dtype=np.float64)
    rmsd = unaligned = rotation = translation = None
    if len(first):
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
            raise ValueError(f"Mapped motif+ligand fixed atom RMSD {rmsd:.6g} A exceeds engineering gate {payload['max_fixed_atom_rmsd']} A")
    return {"sequences": [[c["chain"], "".join(sequences[c["chain"]])] for c in request["chains"]],
            "mapping": mapping, "output_mapping": output_mapping, "motif_mapping": motif_mapping,
            "sampled_lengths": {reconstruction["chain"]: length},
            "motif_mapping_complete": len(motif_mapping) == len(motifs),
            "motif_mapping_requested_count": len(motifs), "motif_mapping_mapped_count": len(motif_mapping),
            "length_semantics": "total_generated_protein_residues_including_motifs",
            "sequence_exclusion_enforcement": ("post_mapping_validation" if motifs
                                                else "native_probability_conditioning_and_validation"),
            "conditioned_coordinate_frame_translation_angstrom": (-centroid).astype(float).tolist(),
            "projection": worker._projection(mapping, len(source)),
            "input_atom_count": len(template), "generated_atom_count": len(output),
            "fixed_atom_rmsd_scope": "only_listed_motif_atoms_plus_fixed_ligand_atoms",
            "fixed_heavy_atom_count": len(source_coords), "fixed_atom_rmsd_angstrom": rmsd,
            "fixed_atom_unaligned_rmsd_angstrom": unaligned,
            "fixed_atom_alignment": {"rotation": rotation, "translation": translation},
            "max_fixed_atom_rmsd_angstrom": payload["max_fixed_atom_rmsd"],
            "fixed_atom_gate": "engineering_tolerance_not_biologically_validated",
            "structure_status": "generated_unrefolded"}
