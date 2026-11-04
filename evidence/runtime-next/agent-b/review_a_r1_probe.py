"""Independent before/after R1 controls; real CPU SciPy, no model/GPU claim."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.abc
import inspect
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--missing-spatial", action="store_true")
    parser.add_argument("--require-lazy", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import numpy as np
    import torch

    if args.missing_spatial:
        class MissingSpatial(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "scipy.spatial":
                    raise ImportError("B controlled unavailable scipy.spatial")
                return None
        sys.meta_path.insert(0, MissingSpatial())

    from astrolabe.app import _load_run_module
    _load_run_module()
    from astrolabe.losses.base import loss_registry
    from astrolabe.losses import design_screening as ds
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    registry = {
        name: {"metadata": dataclasses.asdict(loss_registry.get_metadata(name)),
               "module": loss_registry.get_function(name).__module__,
               "signature": str(inspect.signature(loss_registry.get_function(name)))}
        for name in sorted(loss_registry.valid_losses)
    }
    data = {"head": head, "repo": str(repo), "versions": {"python": sys.version,
            "numpy": np.__version__, "torch": torch.__version__},
            "registry": registry, "registry_count": len(registry),
            "cold_spatial_loaded": "scipy.spatial" in sys.modules,
            "scope": "Real CPU SciPy/PDB mathematical controls, synthetic input coordinates; no pretrained model, GPU or throughput certification."}

    def save():
        args.out.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")

    if args.missing_spatial:
        failures = {}
        for name, call in {
            "clash": lambda: ds._calculate_bindcraft_clash_score("must_not_read.pdb"),
            "hotspot": lambda: ds._bindcraft_hotspot_residues("must_not_read.pdb", binder_chain="B", target_chain="A"),
            "key_residue_plddt": lambda: ds.halludesign_key_res_plddt(
                SimpleNamespace(names=["AB"], complex_paths={"AB": "must_not_read.pdb"}),
                protein_chain="A", ligand_chains="B"),
        }.items():
            try:
                call()
            except ImportError as exc:
                failures[name] = {"type": type(exc).__name__, "message": str(exc),
                                  "cause": type(exc.__cause__).__name__ if exc.__cause__ else None}
            else:
                raise AssertionError((name, "missing dependency was not rejected"))
        data["missing_dependency"] = failures
        assert all(x["message"] == "scipy is required for BindCraft-aligned clash/interface-residue losses."
                   for x in failures.values())
        save()
        return

    from astrolabe.losses import interaction as it, repeat_symmetry as rs
    assert len(registry) == 169, len(registry)
    random.seed(4903)
    np.random.seed(8911)
    torch.manual_seed(10301)

    def rng():
        return {"python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}

    def freeze(value):
        if isinstance(value, np.ndarray):
            return {"dtype": str(value.dtype), "shape": list(value.shape), "values": freeze(value.tolist())}
        if isinstance(value, np.generic):
            return freeze(value.item())
        if isinstance(value, float):
            return {"float_hex": value.hex()}
        if isinstance(value, dict):
            return {str(k): freeze(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [freeze(v) for v in value]
        if isinstance(value, set):
            return freeze(sorted(value))
        return value

    checks = {}
    def check(name, call):
        before = rng()
        try:
            result = call()
            record = {"value": freeze(result)}
        except Exception as exc:
            record = {"exception": {"type": type(exc).__name__, "message": str(exc)}}
        after = rng()
        assert after == before, (name, "global RNG changed")
        checks[name] = record
        return record

    parsed = {"xyz": np.zeros((4, 14, 3)), "mask": np.zeros((4, 14), bool),
              "pdb_idx": [("A", 1), ("A", 2), ("B", 1), ("B", 2)]}
    parsed["xyz"][:, 1] = [[0, 0, 0], [8, 0, 0], [2, 0, 0], [20, 0, 0]]
    parsed["xyz"][:, 4] = [[0.5, 0, 0], [8.5, 0, 0], [2.5, 0, 0], [20.5, 0, 0]]
    parsed["mask"][:, [1, 4]] = True
    data["rng_before_first_geometry"] = rng()
    assert check("first_real_ckdtree", lambda: it._ca_contact_pairs(parsed, "A", "B", cutoff=3.0)) == {"value": [[0, 2]]}
    data["rng_after_first_geometry"] = rng()
    data["spatial_loaded_after_first_geometry"] = "scipy.spatial" in sys.modules
    assert data["spatial_loaded_after_first_geometry"]

    for ca, cb in [(None, None), ("A", "B"), ("A", "Z"), ("A", "A")]:
        for cutoff in [0.0, 2.0, 3.0, 30.0]:
            for method in ["ca", "sc"]:
                check(f"contact-{method}-{ca}-{cb}-{cutoff}",
                      lambda method=method, ca=ca, cb=cb, cutoff=cutoff:
                      getattr(it, f"_{method}_contact_pairs")(parsed, ca, cb, cutoff=cutoff))
    assert checks["contact-ca-A-B-2.0"] == {"value": []}
    assert checks["contact-sc-A-B-2.0"] == {"value": [[0, 2]]}
    for label, variant in {
        "all-masked": dict(parsed, mask=np.zeros_like(parsed["mask"])),
        "malformed-shape": dict(parsed, xyz=np.zeros((2, 3))),
        "nonfinite": dict(parsed, xyz=np.full_like(parsed["xyz"], np.nan)),
    }.items():
        for method in ["ca", "sc"]:
            check(f"{method}-{label}", lambda method=method, variant=variant:
                  getattr(it, f"_{method}_contact_pairs")(variant, "A", "B"))

    atoms = [("CA", "A", 1, (0., 0., 0.), 80.), ("CB", "A", 1, (.4, 0., 0.), 60.),
             ("CA", "A", 2, (4., 0., 0.), 40.), ("CB", "A", 2, (4.4, 0., 0.), 20.),
             ("CA", "B", 1, (2.2, 0., 0.), 90.), ("CB", "B", 1, (2.4, 0., 0.), 70.),
             ("CA", "B", 2, (9., 0., 0.), 50.), ("CA", "C", 1, (3.8, 2., 0.), 30.)]
    lines = [f"ATOM  {n:5d} {atom:^4s} ALA {chain}{residue:4d}    "
             f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00{bf:6.2f}           C"
             for n, (atom, chain, residue, xyz, bf) in enumerate(atoms, 1)]
    fixture = "\n".join(lines + ["TER", "END", ""])
    data["pdb_fixture"] = fixture
    data["pdb_fixture_sha256"] = hashlib.sha256(fixture.encode()).hexdigest()
    from Bio.PDB import PDBParser
    with tempfile.TemporaryDirectory(prefix="review-r1-") as temp:
        pdb = Path(temp) / "geometry.pdb"
        pdb.write_text(fixture)
        model = PDBParser(QUIET=True).get_structure("independent", str(pdb))[0]
        residues_a, residues_b = list(model["A"]), list(model["B"])
        for atom_mode in ["ca", "heavy", "invalid"]:
            for cutoff in [0.0, 2.4, 8.0]:
                check(f"repeat-contact-{atom_mode}-{cutoff}", lambda atom_mode=atom_mode, cutoff=cutoff:
                      rs._residue_pair_contact_count(residues_a, residues_b, cutoff=cutoff, atom_mode=atom_mode))
            check(f"repeat-distance-{atom_mode}", lambda atom_mode=atom_mode:
                  rs._minimum_interchain_atom_distance(residues_a, residues_b, atom_mode=atom_mode))
        check("repeat-contact-empty", lambda: rs._residue_pair_contact_count([], residues_b, cutoff=3., atom_mode="heavy"))
        check("repeat-distance-empty", lambda: rs._minimum_interchain_atom_distance([], residues_b, atom_mode="ca"))
        for cutoff in [0.0, 2.4, 5.0]:
            for only_ca in [False, True]:
                check(f"clash-{cutoff}-{only_ca}", lambda cutoff=cutoff, only_ca=only_ca:
                      ds._calculate_bindcraft_clash_score(pdb, threshold=cutoff, only_ca=only_ca))
            check(f"hotspot-{cutoff}", lambda cutoff=cutoff:
                  ds._bindcraft_hotspot_residues(pdb, binder_chain="B", target_chain="A", atom_distance_cutoff=cutoff))
            check(f"key-residue-{cutoff}", lambda cutoff=cutoff: ds.halludesign_key_res_plddt(
                SimpleNamespace(names=["AB"], complex_paths={"AB": pdb}),
                protein_chain="A", ligand_chains="B", cutoff=cutoff))
        check("interface-residues", lambda: [res.id[1] for res in it._interface_residues_from_structure_model(
            model=model, chain_ids=["A", "B"], interface_distance=2.4, context="B independent review")])

    def alpha(points, **kwargs):
        tri, kept, cutoff = it._alpha_shape_kept_simplex_indices(points, **kwargs)
        return {"points": tri.points, "simplices": tri.simplices, "kept": kept, "cutoff": cutoff}
    tetra = np.asarray([[0., 0., 0.], [3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])
    random_points = np.random.default_rng(8099).normal(size=(12, 3)) * 4.
    for label, points in [("tetra", tetra), ("irregular", random_points)]:
        for radius in [None, 0.1, 10.]:
            check(f"alpha-{label}-{radius}", lambda points=points, radius=radius: alpha(points, alpha_radius=radius))
    for label, points, kwargs in [
        ("duplicates", np.repeat(tetra[:1], 5, axis=0), {}),
        ("nonfinite", np.full((4, 3), np.nan), {}),
        ("bad-radius", tetra, {"alpha_radius": -1.}),
        ("bad-limits", tetra, {"alpha_min_radius": 8., "alpha_max_radius": 3.}),
        ("coplanar-real-qhull-fallback", np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.],[.2,.4,0.]]), {}),
    ]:
        check(f"alpha-{label}", lambda points=points, kwargs=kwargs: alpha(points, **kwargs))
    data["checks"] = checks
    data["check_count"] = len(checks)
    data["rng_after_all_geometry"] = rng()
    assert data["rng_after_all_geometry"] == data["rng_before_first_geometry"]
    assert checks["repeat-contact-ca-2.4"] == {"value": 2}
    assert checks["repeat-distance-empty"] == {"value": {"float_hex": "inf"}}
    data["checks_with_exception"] = [name for name, value in checks.items() if "exception" in value]
    expected_exceptions = {"ca-nonfinite", "sc-nonfinite", "repeat-distance-invalid",
        "repeat-contact-invalid-0.0", "repeat-contact-invalid-2.4", "repeat-contact-invalid-8.0",
        "alpha-tetra-0.1", "alpha-irregular-0.1", "alpha-duplicates", "alpha-nonfinite",
        "alpha-bad-radius", "alpha-bad-limits", "alpha-coplanar-real-qhull-fallback"}
    assert set(data["checks_with_exception"]) == expected_exceptions, data["checks_with_exception"]
    assert all(checks[name]["exception"]["type"] == "ValueError" for name in expected_exceptions)
    import scipy
    import rdkit
    data["versions"].update(scipy=scipy.__version__, rdkit=rdkit.__version__)
    save()
    if args.require_lazy:
        assert not data["cold_spatial_loaded"], "cold full runner eagerly imported scipy.spatial"


if __name__ == "__main__":
    main()
