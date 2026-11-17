"""Real physical-score timings; run from the checkout with PYTHONPATH=.

The native comparison consumes a PDB serialization of the same atoms because
some PyRosetta builds silently import minimal mmCIF as a zero-residue pose.
Rosetta stays optional. Failed native scores are never timing successes.
"""
import argparse
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np
import biotite.structure as struc
from astrolabe.losses.independent import (
    structure_context, independent_sc, independent_cms, polar_context,
    crossgroup_hbond_count, interface_buns,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, default=Path(
        "examples/paper/inputs/assemblies/01_C3_seed2503_step024_reference.cif"))
    parser.add_argument("--chains-a", nargs="+", default=["A"])
    parser.add_argument("--chains-b", nargs="+", default=["B", "C"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--include-rosetta", action="store_true")
    parser.add_argument("--native-polar", action="store_true")
    parser.add_argument("--profile-cms", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    result = SimpleNamespace(complex_paths={"complex": args.structure})
    start = time.perf_counter()
    context = structure_context(result, "complex")
    load_seconds = time.perf_counter() - start
    atoms = context["atoms"]
    a, b = np.isin(atoms.chain_id, args.chains_a), np.isin(atoms.chain_id, args.chains_b)
    assert a.any() and b.any() and not (a & b).any()
    def polar_masks():
        p = polar_context(context)
        return p, np.isin(p["atoms"].chain_id, args.chains_a), np.isin(p["atoms"].chain_id, args.chains_b)
    def hbond():
        p, first, second = polar_masks()
        return crossgroup_hbond_count(p, first, second)
    def buns():
        p, first, second = polar_masks()
        return interface_buns(p, first, second, args.chains_a, args.chains_b)
    functions = {
        "sc_rs": lambda: independent_sc(atoms, a, b),
        "py_contact_ms": lambda: independent_cms(atoms, a, b, distance_weight=0.5),
        "biotite_hbond": hbond, "biotite_buns": buns,
    }
    report = {
        "structure": str(args.structure.resolve()), "sha256": context["digest"].hex(),
        "chains_a": args.chains_a, "chains_b": args.chains_b,
        "heavy_atoms": int(np.count_nonzero(~np.isin(atoms.element, ["H", "D"]))),
        "residues": struc.get_residue_count(atoms), "load_seconds": load_seconds,
        "cpu_count": os.cpu_count(), "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
        "versions": {x: version(x) for x in ("protein-interface", "py-contact-ms", "biotite", "hydride", "rdkit", "numpy")},
        "cms_distance_weight": 0.5, "metrics": {},
    }
    if args.profile_cms:
        from py_contact_ms._core import MolecularSurfaceCalculator, get_radii_from_names
        first, second = atoms[a], atoms[b]
        calc = MolecularSurfaceCalculator()
        calc.add_binder_and_target(
            first.coord, get_radii_from_names(first.res_name.tolist(), first.atom_name.tolist()),
            second.coord, get_radii_from_names(second.res_name.tolist(), second.atom_name.tolist()))
        calc.run.atoms.finalize()
        phases = {}
        for name, action in (
            ("attention", lambda: calc.assign_attention_numbers(calc.run.atoms)),
            ("surface_generation", calc.generate_molecular_surfaces),
            ("dot_finalize", lambda: [x.finalize() for x in calc.run.dots]),
            ("upstream_dense_aggregation", lambda: calc.calc_contact_molecular_surface(target_side=True)),
        ):
            start = time.perf_counter()
            action()
            phases[name] = time.perf_counter() - start
        phases["buried_dots"] = [int(np.count_nonzero(x.buried)) for x in calc.run.dots]
        report["upstream_cms_profile"] = phases
        print("profile", phases, flush=True)
    if args.include_rosetta:
        from biotite.structure.io.pdb import PDBFile
        start = time.perf_counter()
        import pyrosetta
        binary = Path("astrolabe/assets/DAlphaBall.gcc").resolve()
        pyrosetta.init("-mute all -holes:dalphaball " + str(binary))
        report["rosetta_import_init_seconds"] = time.perf_counter() - start
        pdb = PDBFile()
        pdb.set_structure(atoms)
        pose = pyrosetta.Pose()
        pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(pose, str(pdb))
        assert pose.total_residue() == struc.get_residue_count(atoms)
        ns = pyrosetta.rosetta.core.select.residue_selector
        def selector(chains):
            indices = [i for i in range(1, pose.size() + 1) if pose.pdb_info().chain(i) in set(chains)]
            assert indices
            return ns.ResidueIndexSelector(",".join(map(str, indices)))
        first, second = selector(args.chains_a), selector(args.chains_b)
        assert sum(first.apply(pose)) == struc.get_residue_count(atoms[a])
        assert sum(second.apply(pose)) == struc.get_residue_count(atoms[b])
        def native(kind, radii):
            filters = pyrosetta.rosetta.protocols.simple_filters
            cls = filters.ShapeComplementarityFilter if kind == "sc" else filters.ContactMolecularSurfaceFilter
            filt = cls()
            filt.selector1(first)
            filt.selector2(second)
            filt.use_rosetta_radii(radii)
            if kind == "cms":
                filt.distance_weight(0.5)
            value = float(filt.report_sm(pose))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Native {kind} failed: {value}")
            return value
        functions.update({
            "rosetta_sc_default_radii": lambda: native("sc", True),
            "rosetta_sc_historical_radii": lambda: native("sc", False),
            "rosetta_cms_default_radii": lambda: native("cms", True),
            "rosetta_cms_historical_radii": lambda: native("cms", False),
        })
        if args.native_polar:
            from astrolabe.losses.interaction import interface_buried_unsatisfied_hbonds
            from astrolabe.losses.rfd4_feedback import interface_hbond_count
            name = "".join(dict.fromkeys(atoms.chain_id))
            native_result = SimpleNamespace(names=[name], complex_poses=SimpleNamespace(poses={name: pose}))
            functions["rosetta_hbond"] = lambda: interface_hbond_count(
                native_result, args.chains_a, args.chains_b, backend="rosetta")[name]
            functions["rosetta_buns"] = lambda: interface_buried_unsatisfied_hbonds(
                native_result, args.chains_a, args.chains_b, backend="rosetta")[name]
    for name, function in functions.items():
        start = time.perf_counter()
        value = float(function())
        report["metrics"][name] = {"cold_seconds": time.perf_counter() - start, "values": [value], "seconds": []}
        print("cold", name, value, report["metrics"][name]["cold_seconds"], flush=True)
    for repeat in range(args.runs):
        names = list(functions)
        if repeat % 2:
            names.reverse()
        for name in names:
            start = time.perf_counter()
            value = float(functions[name]())
            record = report["metrics"][name]
            record["seconds"].append(time.perf_counter() - start)
            record["values"].append(value)
            print("warm", repeat, name, value, record["seconds"][-1], flush=True)
    for record in report["metrics"].values():
        assert np.isfinite(record["values"]).all()
        record["median_seconds"] = statistics.median(record["seconds"])
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
