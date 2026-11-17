"""Cheap value checks for resolved loss and predocker component settings.

This module deliberately does not import the scoring or docking implementations.
Required/unknown keys and component names belong to startup_validation.
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import re

from astrolabe.loss_config import PHYSICAL_SCORE_BACKENDS, loss_backend, validate_loss_selector


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
        if kind in PHYSICAL_SCORE_BACKENDS:
            backend = None
            try:
                backend = loss_backend(kind, options)
            except ValueError as exc:
                fail("backend", str(exc))
            for key in ("selector_a", "selector_b"):
                if key in options:
                    try:
                        validate_loss_selector(options[key])
                    except ValueError as exc:
                        fail(key, str(exc))
            for key in ("protomer_a", "protomer_b", "binder_chains", "target_chains", "included_protomers"):
                chain(key, multiple=True)
            if options.get("included_protomers") is not None and (
                options.get("protomer_a") is not None or options.get("protomer_b") is not None
            ):
                fail("included_protomers", "conflicts with protomer_a/protomer_b; use one form")
            for key in ("protomer_a_residues", "protomer_b_residues"):
                value = options.get(key)
                if value is None:
                    continue
                if not isinstance(value, str) or not value.strip():
                    fail(key, "expected a nonempty chain-relative residue range or null")
                    continue
                for token in value.split(","):
                    match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", token)
                    if match is None or int(match[1]) < 1 or int(match[2] or match[1]) < int(match[1]):
                        fail(key, f"invalid chain-relative residue range {token!r}")
            for key in ("distance_weight", "buried_sasa_cutoff"):
                number(key, nonnegative=True)
            for key in ("sasa_probe_radius", "separation_distance", "interface_distance", "lower_scale", "upper_scale"):
                number(key, positive=True)
            number("midpoint", positive=True)
            number("midpoint_value", positive=True, maximum=1)
            number("hbond_energy_cutoff")
            number("hydrogen_ph", nonnegative=True, maximum=14)
            number("hbond_distance", positive=True)
            number("hbond_angle", positive=True, maximum=180)
            if options.get("hbond_angle") == 180:
                fail("hbond_angle", "expected an angle below 180 degrees")
            integer("sasa_points")
            if type(options.get("sasa_points")) is int and 0 < options["sasa_points"] < 20:
                fail("sasa_points", "expected at least 20 sphere points")
            if "hydrogens" in options and (not isinstance(options["hydrogens"], str) or options["hydrogens"] not in {"add", "existing"}):
                fail("hydrogens", "expected add or existing")
            if options.get("midpoint_value") == 1:
                fail("midpoint_value", "expected a value below 1")
            for key in ("ignore_surface_res", "strict"):
                if key in options and not isinstance(options[key], bool):
                    fail(key, "expected a boolean")
            if options.get("use_rosetta_radii") is not None and not isinstance(options["use_rosetta_radii"], bool):
                fail("use_rosetta_radii", "expected a boolean or null")
            if backend == "py_contact_ms" and options.get("use_rosetta_radii") is True:
                fail("use_rosetta_radii", "requires backend=rosetta")
            if backend == "biotite" and kind == "interface_buried_unsatisfied_hbonds":
                if options.get("ignore_surface_res") is True:
                    fail("ignore_surface_res", "requires backend=rosetta")
                separation = options.get("separation_distance", 1000.0)
                if finite_number(separation) and separation > 0 and separation != 1000.0:
                    fail("separation_distance", "requires backend=rosetta")
            if backend not in (None, "rosetta") and kind in {"interface_hbond_count", "interface_shape_complementarity"}:
                for key in ("hbond_energy_cutoff", "interface_distance"):
                    if key in options:
                        fail(key, "is a native feedback control; requires backend=rosetta")
        if kind in _CHAIN_PAIRS and kind not in PHYSICAL_SCORE_BACKENDS:
            for key in ("chain_a", "chain_b"):
                chain(key, multiple=True)
            for key in ("protomer_a", "protomer_b"):
                chain(key, multiple=kind in PHYSICAL_SCORE_BACKENDS)
        if kind in {"single_chain_rog", "single_chain_ca_rog"}:
            chain("included_protomers", multiple=True)
        if kind == "secondary_structure_fraction":
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
        if kind == "secondary_structure_fraction" and "sse_type" in options:
            value = options["sse_type"]
            if not isinstance(value, str) or value.upper() not in {"E", "H", "L"}:
                fail("sse_type", "expected E, H, or L")
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
