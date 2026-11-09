"""Typed, local proposal infeasibility; never certifies external failures."""

VDM_INFEASIBILITY_REASONS = frozenset({
    "vdm_site_map_endpoint_not_mutable",
    "vdm_site_map_no_match",
    "vdm_pair_budget_insufficient",
    "vdm_pair_set_insufficient_pairs",
})


class ProposalInfeasible(ValueError):
    """A scientific constraint makes this proposal inadmissible."""

    def __init__(self, reason: str, message: str):
        if reason not in VDM_INFEASIBILITY_REASONS:
            raise ValueError("Unregistered proposal infeasibility reason")
        self.reason = reason
        super().__init__(message)
