"""Audited CPU implementations accepted by the isolated trajectory runtime.

These declarations identify implementations, not claims about an arbitrary
plugin bearing the same name. New GPU-capable components need broker adapters.
"""

# These adapters implement the process/admission contract. This is NOT a
# numerical, cache-equivalence or same-card-concurrency certification.
BROKER_EVALUATORS = {
    'alphafold3': 'af3_resident', 'alphafold2': 'af2_resident',
    'esmfold': 'esmfold_resident', 'esmfold2': 'esmfold2_resident',
    'protenix': 'protenix_resident', 'boltz': 'boltz_resident',
    'chai1': 'chai1_resident', 'opendde': 'opendde_resident',
    'mol1': 'mol1_resident', 'mol-1': 'mol1_resident',
    'rf3': 'rf3_resident', 'rhofold': 'rhofold_resident',
}
CPU_EVALUATORS = frozenset({'dummy', 'rosetta_crystal'})


def declares_broker_protocol(module):
    """Inspect declarations without importing CUDA frameworks in the launcher."""
    import ast
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'evaluators' / (module + '.py')
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return False
    flags = {name.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
             and isinstance(node.value, ast.Constant) for name in node.targets if isinstance(name, ast.Name)}
    return flags.get('BROKER_SHARED_ROOT') is True and flags.get('BROKER_ADMITTED_PROBE') is True

CPU_LOSSES = {
    **{name: 'astrolabe.losses.sequence' for name in (
        'sequence_residue_fraction', 'sequence_region_identity', 'sequence_edge_region_identity')},
    **{name: 'astrolabe.losses.confidence' for name in (
        'plddt', 'ptm', 'iptm', 'pae', 'pde', 'binder_iptm', 'binder_ptm',
        'chain_pair_iptm', 'target_hotspot_plddt', 'ipsae', 'pae_min', 'scaled_pae_min')},
    **{name: 'astrolabe.losses.structure' for name in (
        'single_chain_rog', 'single_chain_ca_rog', 'secondary_structure_fraction',
        'longest_secondary_structure_run_fraction', 'min_dssp_fractions')},
    'protomer_shape_complementarity': 'astrolabe.losses.interaction',
}

# construct_position_selector has a fixed dispatch table. These implementations
# use NumPy, file parsers, DSSP or PyRosetta; they do not launch GPU models.
CPU_SELECTORS = {
    'score': ('score', 'score_position_selector'),
    'sse': ('sse', 'SecondaryStructurePositionSelector'),
    'cdr': ('cdr', 'CDRPositionSelector'),
    'bfactor': ('bfactor', 'BFactorPositionSelector'),
    'interface': ('interface', 'InterfacePositionSelector'),
    'surface': ('surface', 'SurfacePositionSelector'),
    'surface_high_sap': ('protein_quality', 'SurfaceHighSAPPositionSelector'),
    'buried_unsat': ('protein_quality', 'BuriedUnsatPositionSelector'),
    'cavity_neighbor': ('protein_quality', 'CavityNeighborPositionSelector'),
    'ligand_unsat': ('ligand', 'LigandUnsatisfiedPolarPositionSelector'),
    'ligand_contact': ('ligand', 'LigandContactPositionSelector'),
    'membrane_outer_surface': ('ligand', 'MembraneOuterSurfacePositionSelector'),
    'hotspot': ('hotspot', 'HotspotAliasPositionSelector'),
    'long_helix_core': ('sse', 'LongHelixCorePositionSelector'),
    'long_loop': ('sse', 'LongLoopPositionSelector'),
    'rosetta_interface_energy': ('rosetta_interface_energy', 'RosettaInterfaceEnergyPositionSelector'),
}

# Exact source identities for the reviewed fixed dispatch and its implementations.
# Changing either requires a fresh review and an explicit digest update.
SELECTOR_SOURCE_SHA256 = {
    'score': 'e5314c9d50a82475e858506602fd8a7f8ee296534346503b547e47369e427460',
    'utils': '07fa7c38bbfab7858f3dd253e3813385b27e5cb35e71ec85ce96f850257e4a74',
    'sse': '6ac98187f175df20512eaaa81a3b9cb540037b14cb19446abbf7c03ab3cae919',
    'cdr': '4cc16c1f7c959a93ebe6d471c4e618f6cf9e70e9bf4ef5ee6b2f675389211924',
    'bfactor': '32c8beadf78afe376148025997b5e3bf683cd7cebe240e7e06a3db71e9795f40',
    'interface': 'da84c5d26d9377495c45ca9ce4fd3d9bfffdb3e4d71efdfe795ba29988389494',
    'surface': '997542ccdd084af20cc5b4ebf0db157b828e2542620dcd7fd741e428dcf8a7e6',
    'protein_quality': '15e0b10273fa03f8095f5f0d60fa6387df62990679698a04573b30bbf697e502',
    'ligand': 'da72d8d091bbdd766a9a6994ebce02548caec68b21e0e1a3dfdef4933844f6ab',
    'hotspot': '9fd0abf8b43b7d174c7e36bb811ead8f511edbe11a8a5299255be9f6351e3a63',
    'rosetta_interface_energy': '64625bf1dd7e1549ce4839e06ec53eaf58e63b85a429604536bd7fe1de5ec1a1',
}
