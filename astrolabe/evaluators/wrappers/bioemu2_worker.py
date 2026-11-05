"""Standalone subprocess entrypoint for the separately installed BioEmu2."""
import json
import os
from pathlib import Path
import sys


def main(request_path):
    request = json.loads(Path(request_path).read_text())
    output = Path(request["output_dir"])
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from bioemu2 import (EmuSession, Protein, System, check_physicality, get_msa,
                         load_model, map_cif_atom_order, save_frames)
    from bioemu2.trajectory_io import save_mdtraj_cif_frame

    chains = request["chains"]
    system = System([Protein(sequence=c["sequence"], asym_ids=[c["id"]]) for c in chains])
    msa_paths = {}
    for index, chain in enumerate(chains):
        path = output / f"query_{index}.a3m"
        path.write_text(f">query\n{chain['sequence']}\n")
        msa_paths[chain["id"]] = path
    msa = get_msa(system, source=msa_paths)
    model = load_model(checkpoint="bioemu2-preview", cache_dir=Path(request["cache_dir"]),
                       device=request["device"], precision=request["precision"],
                       matmul_precision=request["matmul_precision"])
    mode = request["mode"]
    session = EmuSession(system, model=model, mode="equilibrium" if mode == "propagation" else mode, msa=msa)
    if mode == "propagation":
        start = map_cif_atom_order(system, request["initial_cif"])
        trajectory = start[[0] * request["num_samples"]]
        for step in range(request["propagation_steps"]):
            trajectory = session.propagate(trajectory, lag_time_ns=request["lag_time_ns"],
                batch_size=request["batch_size"], sampler=request["sampler"],
                seed=request["seed"] + step if request["seed"] is not None else None, progress=False)
    else:
        trajectory = session.sample(num_samples=request["num_samples"], batch_size=request["batch_size"],
                                    sampler=request["sampler"], seed=request["seed"], progress=False)
    physicality = check_physicality(trajectory, system=system)
    save_frames(trajectory, output, physicality=physicality)
    passed = [i for i, valid in enumerate(physicality.passed) if valid]
    (output / "result.json").write_text(json.dumps({"representative_index": passed[0] if passed else None,
        "num_samples": trajectory.n_frames, "physicality_passed": len(passed)}, indent=2) + "\n")
    if not passed:
        raise RuntimeError("No BioEmu2 frame passed physicality; see physicality.json")
    save_mdtraj_cif_frame(trajectory[passed[0]], output / "representative.cif")


if __name__ == "__main__":
    main(sys.argv[1])
