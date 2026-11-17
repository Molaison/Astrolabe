import math
import numpy as np
import pyrosetta
from pyrosetta.rosetta.core.pose import get_chain_from_chain_id

from astrolabe.evaluators.utils import ComplexPoses
from astrolabe.selectors.base import PositionSelector


class SurfacePositionSelector(PositionSelector):
    def __init__(self, complex_poses: ComplexPoses, merge="any", *,
                 core_cutoff=None, surface_cutoff=None):
        # Explicit rolling-ball cutoffs are SASA in A^2, not neighbor counts.
        # None preserves the current native LayerSelector defaults for other tasks.
        if (core_cutoff is None) != (surface_cutoff is None):
            raise ValueError("Specify both core_cutoff and surface_cutoff, or neither.")
        if core_cutoff is not None:
            if (isinstance(core_cutoff, bool) or isinstance(surface_cutoff, bool)
                    or not math.isfinite(core_cutoff) or not math.isfinite(surface_cutoff)
                    or not 0 <= core_cutoff < surface_cutoff):
                raise ValueError("Surface SASA cutoffs must be finite with 0 <= core < surface.")
            core_cutoff, surface_cutoff = float(core_cutoff), float(surface_cutoff)
        self.core_cutoff, self.surface_cutoff = core_cutoff, surface_cutoff
        super().__init__(complex_poses, merge=merge)

    def select_positions(self):
        self.sites = self.surface_selector(
            self.complex_poses.poses
        )
        return self.sites

    def surface_selector(self, poses, ):
        surface_sele = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
        surface_sele.set_layers(False, False, True)
        surface_sele.set_use_sc_neighbors(False)
        if self.core_cutoff is not None:
            surface_sele.set_cutoffs(self.core_cutoff, self.surface_cutoff)
        selected_sites = {}
        for oligo_name, pose in poses.items():
            for chain in pose.split_by_chain():
                chain_name = get_chain_from_chain_id(chain.chain(1), chain)
                selected_sites[chain_name] = np.array(surface_sele.apply(chain))
        return selected_sites
