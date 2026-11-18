"""Mutator configuration rules shared by preflight and construction."""

from collections.abc import Mapping


MUTATOR_DEFAULTS = {
    "rfd4": ("rfd4",),
    "mcmc": ("MCMC",), "pore_blocker_mcmc": ("MCMC",),
    "ligandmpnn": ("LigandMPNN",), "grouped_ligandmpnn": ("LigandMPNN",),
    "proteinmpnn": ("LigandMPNN",), "multistateproteinmpnn": ("LigandMPNN",),
    "multistate_proteinmpnn": ("LigandMPNN",),
    "faraparttiedmultitargetligandmpnn": ("LigandMPNN",),
    "far_apart_tied_multitarget_ligandmpnn": ("LigandMPNN",),
    "proton_pottsmpnn": ("ProtonPottsMPNN",), "proton-pottsmpnn": ("ProtonPottsMPNN",),
    "vdm_pair": ("VDMPair",), "vdm_proton": ("ProtonPottsMPNN",),
    "dynamicmpnn": ("DynamicMPNN",), "surfdesign": ("SurfDesign",),
    "surfpro": ("SurfPro",), "carbondesign": ("CarbonDesign",),
    "dplm": ("DPLM",), "nampnn": ("NAMPNN",),
}


def mutator_position_selector(alias, kind, options, selectors):
    def selected(value, path):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: expected a non-empty selector expression string or null")
        return value, path

    override = options.get("position_selector")
    if override is not None:
        return selected(override, f"mutator.mutator_terms.{alias}.position_selector")
    if selectors is None:
        return None, None
    if not isinstance(selectors, Mapping):
        raise ValueError("mutator.selectors: expected a mapping from alias or type to selector expression")
    for key in (alias, kind, kind.lower()):
        selector = selectors.get(key)
        if selector is not None:
            return selected(selector, f"mutator.selectors.{key}")
    return None, None
