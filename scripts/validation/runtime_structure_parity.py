"""Compare emitted structures using error observed in repeated direct runs.

Checks the coordinates actually delivered by the workflow. PDB text precision
is not a certificate for internal model tensors; all native candidates and
cache-dependent intermediate tensors require separate backend evidence.
"""
from pathlib import Path
import json
import math


def collect(directory):
    result = {}
    for provenance in sorted(Path(directory).glob("results/**/*_run/provenance.json")):
        record = json.loads(provenance.read_text())
        identity = str((record.get("run", {}).get("repeat_index"), record.get("run", {}).get("seed")))
        if record.get("state") != "completed":
            continue
        for path in sorted((provenance.parent / "structures").rglob("*")):
            if path.is_file() and path.suffix.lower() in (".pdb", ".cif", ".mmcif"):
                key = identity + ":" + path.relative_to(provenance.parent).as_posix()
                if key in result:
                    raise ValueError("duplicate structure identity: " + key)
                result[key] = str(path)
    return result


def atoms(path):
    path = Path(path)
    result = {}
    if path.suffix.lower() == ".pdb":
        model = "1"
        with path.open() as stream:
            for line in stream:
                if line.startswith("MODEL "):
                    model = line[10:14].strip()
                if line[:6] not in ("ATOM  ", "HETATM"):
                    continue
                key = (model, line[21:22], line[22:27], line[17:20], line[12:16], line[16:17], line[:6])
                if key in result:
                    raise ValueError("duplicate delivered atom identity: " + str(key))
                result[key] = tuple(float(line[start:start+8]) for start in (30, 38, 46))
    else:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        content = MMCIF2Dict(str(path))
        xyz = [content["_atom_site.Cartn_" + axis] for axis in "xyz"]
        total = len(xyz[0])
        keys = ["pdbx_PDB_model_num", "label_asym_id", "label_seq_id", "auth_seq_id",
                "pdbx_PDB_ins_code", "label_comp_id", "label_atom_id", "label_alt_id", "group_PDB"]
        columns = [content.get("_atom_site." + key, ["?"] * total) for key in keys]
        for index in range(total):
            key = tuple(column[index] for column in columns)
            if key in result:
                raise ValueError("duplicate delivered atom identity: " + str(key))
            result[key] = tuple(float(column[index]) for column in xyz)
    if not result or any(not math.isfinite(x) for xyz in result.values() for x in xyz):
        raise ValueError("missing or nonfinite delivered coordinates: " + str(path))
    return result


def compare(records):
    direct = [r for r in records if r["mode"] == "direct" and r["valid_workload"]]
    if len(direct) < 2:
        return {"certified": False, "reason": "need at least two successful direct structure repeats"}
    reference = direct[0].get("structure_artifacts", {})
    if not any(r.get("structure_artifacts") for r in records):
        return {"certified": None, "reason": "workflow emitted no structure artifacts; model coordinate parity not measured"}
    differences, checked, largest_tolerance = [], 0, 0.0
    for run in records:
        if run["valid_workload"] and run.get("structure_artifacts", {}).keys() != reference.keys():
            differences.append({"run": run["run"], "kind": "structure_artifact_identity"})
    if differences:
        return {"certified": False, "differences": differences}
    for name, reference_path in reference.items():
        expected = atoms(reference_path)
        tolerances = {key: [0.0, 0.0, 0.0] for key in expected}
        for trial in direct[1:]:
            observed = atoms(trial["structure_artifacts"][name])
            if observed.keys() != expected.keys():
                return {"certified": False, "reason": "direct atom identities are unstable", "artifact": name}
            for key, coordinates in expected.items():
                tolerances[key] = [max(tolerances[key][axis], abs(coordinates[axis] - observed[key][axis])) for axis in range(3)]
        largest_tolerance = max(largest_tolerance, max(max(value) for value in tolerances.values()))
        for run in records:
            if not run["valid_workload"] or run["mode"] == "direct":
                continue
            observed = atoms(run["structure_artifacts"][name])
            if observed.keys() != expected.keys():
                differences.append({"run": run["run"], "kind": "atom_identity", "artifact": name})
                continue
            changed = sum(any(abs(x - y) > tol for x, y, tol in zip(expected[key], observed[key], tolerances[key])) for key in expected)
            if changed:
                differences.append({"run": run["run"], "kind": "coordinates", "artifact": name, "atoms_outside_empirical_tolerance": changed})
            checked += len(expected)
    return {"certified": not differences, "compared_atoms": checked, "artifacts_per_trajectory_set": len(reference),
            "largest_empirical_coordinate_tolerance_angstrom": largest_tolerance,
            "tolerance_source": "per-atom/per-axis maximum absolute deviation from the first direct emitted structure",
            "scope": "delivered PDB/mmCIF coordinates and atom identities, at file precision; internal tensors and unexported candidates are not certified",
            "differences": differences}
