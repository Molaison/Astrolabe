#!/usr/bin/env python3
"""Compare two real, seeded ProteinMPNN calls with a resident model owner."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def daemon_factory(settings):
    import torch
    from astrolabe.model_execution import kernel_handler
    from astrolabe.mutators.native_daemons import _load_ligandmpnn

    checkpoint_reads = []
    original_load = torch.load

    def counted_load(path, *args, **kwargs):
        checkpoint_reads.append(str(path))
        return original_load(path, *args, **kwargs)

    def loader(config):
        with patch.object(torch, "load", counted_load):
            infer, metadata = _load_ligandmpnn(config)
        runner = inspect.getclosurevars(infer).nonlocals["runner"]

        def predict(payload):
            with patch.object(torch, "load", counted_load):
                output = infer(payload)
            return {"output": output, "checkpoint_reads": list(checkpoint_reads),
                    "model_token": id(runner.model), "pid": os.getpid(),
                    "device": str(runner.device)}

        return predict, metadata

    return kernel_handler(settings, loader, torch_device="auto")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    paths = ("astrolabe/model_execution.py", "astrolabe/model_daemon.py",
             "astrolabe/mutators/native_daemons.py",
             "astrolabe/mutators/wrappers/ligandmpnn_run.py")
    return {path: digest(ROOT / path) for path in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=ROOT / "examples/input/9TP_cid.cif")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="Retry an incomplete check in the same output directory.")
    args = parser.parse_args(argv)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import numpy as np
    import torch
    from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
    from omegaconf import OmegaConf
    from astrolabe.model_execution import NativeModelClient, capture_random_state
    from astrolabe.mutators.wrappers.ligandmpnn_run import LigandMPNN

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("The CUDA check requires a GPU in the current allocation.")
    if args.seed == 0:
        parser.error("This deterministic comparison needs a nonzero seed.")
    checkpoint = args.weights.resolve() / "proteinmpnn_v_48_020.pt"
    if not checkpoint.is_file():
        parser.error(f"Missing ProteinMPNN checkpoint: {checkpoint}")
    out = (args.out_dir or Path(tempfile.mkdtemp(prefix="astrolabe-mpnn-check-"))).resolve()
    if args.out_dir:
        if (out / "verification.json").exists():
            parser.error(f"A completed verification already exists at {out}")
        out.mkdir(parents=True, exist_ok=args.resume)
    before = source_identity()
    source = args.input.resolve()
    structure_parser = MMCIFParser(QUIET=True) if source.suffix == ".cif" else PDBParser(QUIET=True)
    structure = structure_parser.get_structure("check", str(source))

    class ChainSelection(Select):
        def accept_chain(self, chain):
            return chain.id == args.chain

    pdb_path = out / "input.pdb"
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(str(pdb_path), select=ChainSelection())
    if not any(line.startswith("ATOM") for line in pdb_path.read_text().splitlines()):
        parser.error(f"No protein atoms found for chain {args.chain!r}")

    cfg = OmegaConf.merge(OmegaConf.load(ROOT / "config/env/env.yaml"),
                          OmegaConf.load(ROOT / "config/base.yaml"))
    cfg.seed = args.seed
    cfg.exclude_AA = ""
    cfg.weight_path.ligandmpnn = str(args.weights.resolve())
    cfg.mutator.LigandMPNN.model_type = "protein_mpnn"
    cfg.mutator.LigandMPNN.out_folder = str(out / "direct")
    config = OmegaConf.to_container(cfg.mutator.LigandMPNN, resolve=True)
    features = {"pdb_path": str(pdb_path), "chains_to_design": args.chain,
                "parse_these_chains_only": args.chain, "fixed_residues": ""}

    def reset():
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    reset()
    direct = LigandMPNN(OmegaConf.create(config))
    reference = [direct.design_sequences(**features) for _ in range(2)]
    reference_random_state = capture_random_state()
    del direct
    reset()
    config["out_folder"] = str(out / "daemon")
    client = NativeModelClient("ligandmpnn",
                               "scripts.maintenance.check_resident_mpnn:daemon_factory",
                               {"runner_config": config}, cwd=str(ROOT))
    try:
        results = [client.request({"features": features}) for _ in range(2)]
        daemon_random_state = capture_random_state()
        owner = client.owner.status()
    finally:
        client.shutdown()
    numeric_keys = ("S", "sampling_probs", "log_probs", "decoding_order")
    hashes = []
    for expected, actual in zip(reference, results):
        sample_hashes = {}
        for key in numeric_keys:
            left = np.asarray(expected[key].detach().cpu())
            right = np.asarray(actual["output"][key].detach().cpu())
            if left.dtype != right.dtype or not np.array_equal(left, right, equal_nan=True):
                raise AssertionError(f"Direct and daemon {key} differ")
            sample_hashes[key] = hashlib.sha256(left.tobytes()).hexdigest()
        hashes.append(sample_hashes)
    if json.dumps(reference_random_state, sort_keys=True) != json.dumps(daemon_random_state, sort_keys=True):
        raise AssertionError("The caller's random-state continuation differs")
    if len({item["model_token"] for item in results}) != 1 or len({item["pid"] for item in results}) != 1:
        raise AssertionError("The loaded model owner changed between requests")
    if any(item["checkpoint_reads"] != [str(checkpoint)] for item in results):
        raise AssertionError("The checkpoint did not load exactly once")
    if before != source_identity():
        raise RuntimeError("Source changed during verification; repeat on the completed change")
    report = {"verified": True, "device": results[0]["device"], "seed": args.seed,
              "requests": len(results), "checkpoint_reads": results[-1]["checkpoint_reads"],
              "checkpoint_sha256": digest(checkpoint), "input_sha256": digest(source),
              "source_sha256": before, "sample_sha256": hashes,
              "owner_pid": results[0]["pid"], "model_token": results[0]["model_token"],
              "owner_identity": owner["identity"], "caller_random_state_equal": True,
              "performance_measured": False}
    report_path = out / "verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
