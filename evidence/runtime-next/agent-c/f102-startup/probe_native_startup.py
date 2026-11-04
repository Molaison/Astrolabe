#!/usr/bin/env python3
"""Independent, bounded real-PyRosetta correctness probe; no timing claim.

Use two fresh processes with the identical frozen protocol, one per exact HEAD.
No scientific RNG state is restored and no native symbol is replaced.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--expect-head", required=True)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.repo = args.repo.resolve()
    args.output = args.output.resolve()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    assert args.expect_head in (protocol["baseline_commit"], protocol["candidate_commit"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "requires an explicitly GPU-hidden CPU environment"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ.get(name) == "1", f"requires {name}=1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    cpu_budget = int(protocol["budgets"]["cpu_seconds_per_process"])
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_budget, cpu_budget + 1))

    def expired(signum, _frame):
        raise TimeoutError(f"native correctness budget reached, signal={signum}")

    signal.signal(signal.SIGXCPU, expired)
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(int(protocol["budgets"]["wall_seconds_per_process"]))
    start_wall, start_cpu = time.monotonic(), time.process_time()
    result = {
        "schema": "astrolabe-startup-native-scientific-result-v1",
        "status": "PARTIAL", "native": True, "protocol": protocol,
        "protocol_sha256": sha(protocol_bytes),
        "probe_sha256": sha(Path(__file__).read_bytes()),
        "argv": sys.argv,
        "python": sys.version,
        "rng": {}, "checks": {},
    }

    def checkpoint():
        result["observed_wall_seconds"] = time.monotonic() - start_wall
        result["observed_cpu_seconds"] = time.process_time() - start_cpu
        write(args.output, result)

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True)
        result["source"] = {"root": str(args.repo), "head": head, "tracked_status": status,
            "files": {p: sha((args.repo / p).read_bytes()) for p in (
                "astrolabe/losses/interaction.py", "astrolabe/runtime_requirements.py",
                "astrolabe/losses/utils.py", "astrolabe/losses/base.py", "astrolabe/argparser.py",
                "tests/test_per_residue_rosetta_scores.py", "tests/test_protein_quality_selectors_pyrosetta.py")}}
        assert head == args.expect_head and not status, "exact clean source identity required"
        import random
        import numpy as np
        import torch
        torch.set_num_threads(1)
        result["dependencies"] = {"numpy": np.__version__, "torch": torch.__version__}

        def state():
            n = np.random.get_state()
            return {
                "python": random.getstate(),
                "numpy": {"algorithm": n[0], "keys_base64": base64.b64encode(n[1].tobytes()).decode(),
                    "dtype": str(n[1].dtype), "pos": n[2], "has_gauss": n[3], "cached_gaussian": n[4]},
                "torch_cpu_base64": base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode(),
            }

        def mark(label):
            # JSON roundtrip normalizes tuples without altering any RNG.
            result["rng"][label] = json.loads(json.dumps(state()))
            checkpoint()

        seed = int(protocol["python_numpy_torch_seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        mark("before_cold_import_diagnostic")
        from astrolabe.app import _load_run_module
        _load_run_module()
        import astrolabe.losses.interaction as interaction
        from astrolabe.losses.base import loss_registry
        from astrolabe.runtime_requirements import _pyrosetta_loss_types_from_source, task_requirements
        mark("after_cold_import_diagnostic")
        result["cold_modules"] = {"pyrosetta": "pyrosetta" in sys.modules,
            "scipy.spatial": "scipy.spatial" in sys.modules}
        from astrolabe.argparser import set_random_seeds
        set_random_seeds(seed)
        mark("production_seeded")
        result["registry_order"] = loss_registry.valid_losses
        result["registry"] = {name: {
            "metadata": dataclasses.asdict(loss_registry.get_metadata(name)),
            "signature": str(inspect.signature(loss_registry.get_function(name))),
            "module": loss_registry.get_function(name).__module__,
            "qualname": loss_registry.get_function(name).__qualname__,
        } for name in loss_registry.valid_losses}
        result["pyrosetta_loss_classification"] = sorted(_pyrosetta_loss_types_from_source())
        result["requirements"] = {name: task_requirements({"loss": {"loss_terms": {
            "control": {"_type": name}}}}) for name in loss_registry.valid_losses}
        mark("after_registry_and_requirements")
        interaction._require_pyrosetta()
        mark("after_first_require")
        pr = interaction.pyrosetta
        assert pr is not None and pr is sys.modules["pyrosetta"]
        from astrolabe.utils.pyrosetta_init import pyrosetta_was_initialized
        from pyrosetta.rosetta.core.scoring import hbonds
        from pyrosetta.rosetta.core.simple_metrics.per_residue_metrics import PerResidueSasaMetric
        from pyrosetta.rosetta.protocols.simple_filters import (
            ContactMolecularSurfaceFilter, HolesFilter, ShapeComplementarityFilter)
        from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
        actual = {"pyrosetta": pr, "rosetta_hbonds": hbonds, "PerResidueSasaMetric": PerResidueSasaMetric,
            "ContactMolecularSurfaceFilter": ContactMolecularSurfaceFilter, "HolesFilter": HolesFilter,
            "ShapeComplementarityFilter": ShapeComplementarityFilter, "MutateResidue": MutateResidue}
        result["native_symbols"] = {k: {"complete_identity": getattr(interaction, k) is v,
            "name": getattr(v, "__name__", None), "module": getattr(v, "__module__", None)} for k, v in actual.items()}
        result["native_version"] = str(pr.version())
        result["initialized_before_explicit_init"] = pyrosetta_was_initialized(pr)
        assert not result["initialized_before_explicit_init"], "import guard must not initialize Rosetta"
        pr.init(protocol["native_initialization_options"])
        result["initialized_after_explicit_init"] = pyrosetta_was_initialized(pr)
        mark("after_native_init")

        reference = pr.pose_from_sequence(protocol["sequence"])
        pose = None
        for translation_values in protocol["chain_translations_angstrom"]:
            part = reference.clone()
            translation = pr.rosetta.numeric.xyzVector_double_t(*translation_values)
            for ri in range(1, part.total_residue() + 1):
                for ai in range(1, part.residue(ri).natoms() + 1):
                    atom_id = pr.rosetta.core.id.AtomID(ai, ri)
                    part.set_xyz(atom_id, part.xyz(atom_id) + translation)
            if pose is None:
                pose = part
            else:
                pr.rosetta.core.pose.append_pose_to_pose(pose, part, True)
        assert pose.total_residue() == protocol["expected_residue_count"]
        assert pose.num_chains() == len(protocol["chain_order"])
        pose.pdb_info(pr.rosetta.core.pose.PDBInfo(pose))
        for chain_index, chain_id in enumerate(protocol["chain_order"], 1):
            for ri in range(pose.chain_begin(chain_index), pose.chain_end(chain_index) + 1):
                pose.pdb_info().chain(ri, chain_id)
        pose.update_residue_neighbors()

        def geometry():
            return [{"residue": ri, "chain": pose.pdb_info().chain(ri), "name3": pose.residue(ri).name3(),
                "atoms": [{"name": pose.residue(ri).atom_name(ai),
                    "xyz": [pose.residue(ri).xyz(ai).x, pose.residue(ri).xyz(ai).y, pose.residue(ri).xyz(ai).z]}
                    for ai in range(1, pose.residue(ri).natoms() + 1)]}
                for ri in range(1, pose.total_residue() + 1)]

        result["geometry_before"] = geometry()
        mark("after_pose_build")
        from types import SimpleNamespace
        eval_result = SimpleNamespace(names=["ABC"], complex_poses=SimpleNamespace(poses={"ABC": pose}))
        result["metric_values"] = []
        result["registered_loss_values"] = []
        for _ in range(int(protocol["repeat_loss_calls"])):
            sasa = interaction.sasa_calculator(pose)
            assert sasa.shape == (protocol["expected_residue_count"],) and np.all(np.isfinite(sasa))
            assert float(np.sum(sasa)) > 0, "native metric must perform nonempty scientific work"
            result["metric_values"].append(sasa.tolist())
            values = {name: {k: float(v) for k, v in loss_registry.get_function(name)(eval_result, **kw).items()}
                for name, kw in protocol["registered_losses"].items()}
            assert all(set(v) == {"ABC"} and np.isfinite(v["ABC"]) for v in values.values())
            assert values["com_distance"]["ABC"] > 0
            result["registered_loss_values"].append(values)
        result["geometry_after"] = geometry()
        mark("after_native_losses")
        result["cuda_initialized"] = torch.cuda.is_initialized()
        result["checks"] = {
            "native_symbols_complete": all(v["complete_identity"] for v in result["native_symbols"].values()),
            "first_require_preserves_rng": result["rng"]["after_first_require"] == result["rng"]["after_registry_and_requirements"],
            "registry_requirements_preserve_rng": result["rng"]["production_seeded"] == result["rng"]["after_registry_and_requirements"],
            "losses_preserve_rng": result["rng"]["after_pose_build"] == result["rng"]["after_native_losses"],
            "losses_preserve_geometry": result["geometry_before"] == result["geometry_after"],
            "repeated_native_metric_exact": result["metric_values"][0] == result["metric_values"][1],
            "repeated_registered_losses_exact": result["registered_loss_values"][0] == result["registered_loss_values"][1],
            "cpu_only": not result["cuda_initialized"],
            "native_initialization_valid": result["initialized_after_explicit_init"],
        }
        assert all(result["checks"].values()), result["checks"]
        result["status"] = "PASS"
    except BaseException as exc:
        result["status"] = "FAIL"
        result["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        signal.alarm(0)
        checkpoint()
    print(json.dumps({"status": result["status"], "output": str(args.output),
        "registry_count": len(result.get("registry", {})), "checks": result["checks"],
        "observed_cpu_seconds": result["observed_cpu_seconds"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
