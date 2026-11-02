"""Audited CPU implementations accepted by the isolated trajectory runtime.

These declarations identify implementations, not claims about an arbitrary
plugin bearing the same name. New GPU-capable components need broker adapters.
"""

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
