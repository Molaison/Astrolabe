"""Cheap value checks for resolved loss and predocker component settings.

This module deliberately does not import the scoring or docking implementations.
Required/unknown keys and component names belong to startup_validation.
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil


_INTERFACE_PAE = {"ipae", "ipae_min", "ipae_minimal", "ipae_max", "ipsae"}
_CHAIN_PAIRS = _INTERFACE_PAE | {
    "protomer_shape_complementarity", "interface_buried_unsatisfied_hbonds",
    "interface_holes", "binder_pair_tmscore",
}
_AUTODOCK = {"autodock_vina", "autodock_gpu"}
_HADDOCK = {"haddock3", "haddock3_nucleic", "HADDOCK3_nucleic"}


def validate_scoring_settings(family, kind, options, *, path, check_paths) -> list[str]:
    """Return field-qualified errors for common resolved loss/predocker values.

    ``check_paths`` is for definite local assets only; no model/framework is
    imported and optional null defaults are not interpreted as missing assets.
    """
    if family not in {"loss", "predocker"} or not isinstance(options, dict):
        return []
    errors: list[str] = []

    def fail(key, message):
        errors.append(f"{path}.{key}: {message}")

    def finite_number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(value)
        except (OverflowError, TypeError, ValueError):
            return False

    def number(key, *, positive=False, nonnegative=False, maximum=None, nullable=False):
        if key not in options or (nullable and options[key] is None):
            return
        value = options[key]
        if not finite_number(value):
            fail(key, "expected a finite number")
        elif positive and value <= 0:
            fail(key, "expected a positive number")
        elif nonnegative and value < 0:
            fail(key, "expected a nonnegative number")
        elif maximum is not None and value > maximum:
            fail(key, f"expected a number <= {maximum}")

    def integer(key, *, positive=True, nullable=False):
        if key not in options or (nullable and options[key] is None):
            return
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0):
            fail(key, "expected a positive integer" if positive else "expected an integer")

    def chain(key, *, multiple=False):
        if key not in options or options[key] is None:
            return
        value = options[key]
        if isinstance(value, str):
            good = bool(value.strip()) and (not multiple or all(x.strip() for x in value.split(",")))
        elif multiple and isinstance(value, (list, tuple)):
            good = bool(value) and all(isinstance(x, str) and x.strip() for x in value)
        else:
            good = False
        if not good:
            fail(key, "expected a non-empty chain selector" + (" or list of selectors" if multiple else ""))

    def asset(key, *, nullable=True):
        if key not in options or (nullable and options[key] is None):
            return
        value = options[key]
        if not isinstance(value, str) or not value.strip():
            fail(key, "expected a non-empty path or command")
        elif check_paths and "${" not in value and not (Path(value).expanduser().exists() or shutil.which(value)):
            fail(key, "asset does not exist")

    if family == "loss":
        if kind in _CHAIN_PAIRS:
            for key in ("chain_a", "chain_b"):
                chain(key, multiple=True)
            for key in ("protomer_a", "protomer_b"):
                chain(key, multiple=kind in {"interface_holes", "interface_buried_unsatisfied_hbonds"})
        if kind in {"single_chain_rog", "single_chain_ca_rog"}:
            chain("included_protomers", multiple=True)
        if kind in {"secondary_structure_fraction", "protomer_shape_complementarity", "interface_buried_unsatisfied_hbonds"}:
            chain("included_protomers", multiple=False)
        if kind == "binder_pair_tmscore":
            for key in ("oligo_a", "oligo_b", "binder_chain"):
                chain(key)
        if kind in _INTERFACE_PAE:
            for key in ("sc_cutoff", "ca_cutoff"):
                number(key, positive=True)
            if "contact_method" in options:
                value = options["contact_method"]
                if not isinstance(value, str) or value.lower().replace("-", "_") not in {"sc", "ca", "ca_ca", "c_alpha", "calpha"}:
                    fail("contact_method", "expected sc or a CA method")
        if kind == "ipsae":
            number("pae_cutoff", positive=True)
        if kind == "interface_holes":
            number("interface_distance", positive=True)
        if kind == "interface_buried_unsatisfied_hbonds":
            number("separation_distance", positive=True)
            if "ignore_surface_res" in options and not isinstance(options["ignore_surface_res"], bool):
                fail("ignore_surface_res", "expected a boolean")
        if kind == "secondary_structure_fraction" and "sse_type" in options:
            value = options["sse_type"]
            if not isinstance(value, str) or value.upper() not in {"E", "H", "L"}:
                fail("sse_type", "expected E, H, or L")
        if kind == "protomer_shape_complementarity" and "strict" in options and not isinstance(options["strict"], bool):
            fail("strict", "expected a boolean")
        return errors

    if kind in _AUTODOCK:
        for key in ("num_modes", "exhaustiveness", "num_samples"):
            integer(key)
        number("binding_site_coverage", nonnegative=True, maximum=100)
        for key in ("binding_site_padding", "reference_box_padding"):
            number(key, nonnegative=True)
        # The AutoDock reference box clamps this floor with max(value, 0.0).
        # Zero and negative values therefore have the same supported effect.
        number("reference_min_box_size")
        for key in ("obabel_path", "vina_path", "autodock_gpu_path", "autogrid4_path", "mk_prepare_receptor", "mk_prepare_ligand", "mk_prepare_python_bin", "reference_structure_path"):
            asset(key, nullable=key != "obabel_path")
    elif kind in _HADDOCK:
        for key in ("sampling", "ncores", "max_restraints", "max_binding_site_pairs"):
            integer(key)
        integer("max_anchor_chains", nullable=True)
        for key in ("binder_chain",):
            chain(key)
        for key in ("target_chains", "anchor_chains"):
            chain(key, multiple=True)
        for key in ("contact_cutoff", "binding_site_max_distance"):
            number(key, positive=True)
        number("binding_site_min_distance", nonnegative=True)
        if "restraint_distance" in options and options["restraint_distance"] is not None:
            value = options["restraint_distance"]
            if isinstance(value, str):
                try:
                    value = [float(part) for part in value.replace(",", " ").split()]
                except ValueError:
                    value = []
            if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(finite_number(x) for x in value) or not (0 < value[0] <= value[1] <= value[2]):
                fail("restraint_distance", "expected three ordered positive distances")
        asset("haddock3_bin")
    elif kind == "symmetry_canonicalizer" and options.get("backend") == "rpxdock":
        integer("rpxdock_timeout_s")
        for key in ("rpxdock_python_bin", "rpxdock_vendor_repo", "rpxdock_willutil_repo"):
            asset(key)
    return errors
