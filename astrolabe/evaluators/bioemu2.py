"""BioEmu2 evaluator, executed in its own Python environment."""
from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile

from astrolabe.core.folding_input import ProteinChain
from astrolabe.evaluators.base import Evaluator
from astrolabe.evaluators.utils import StandardScores


class BioEmu2Evaluator(Evaluator):
    def __init__(self, *, output_dir, python_bin, cache_dir, mode="equilibrium",
                 num_samples=1, batch_size=1, sampler="edm", seed=None,
                 initial_cif=None, lag_time_ns=1.0, propagation_steps=1,
                 device="cuda", precision="bf16-mixed", matmul_precision="high"):
        from astrolabe.backend_startup import validate_backend_settings

        errors = validate_backend_settings(
            "evaluator", "bioemu2",
            dict(python_bin=python_bin, cache_dir=cache_dir, mode=mode,
                 num_samples=num_samples, batch_size=batch_size, sampler=sampler,
                 seed=seed, initial_cif=initial_cif, lag_time_ns=lag_time_ns,
                 propagation_steps=propagation_steps, device=device,
                 precision=precision, matmul_precision=matmul_precision),
            path="evaluator.bioemu2", check_paths=True,
        )
        if errors:
            raise ValueError("\n".join(errors))
        if mode not in {"equilibrium", "structure", "propagation"}:
            raise ValueError("BioEmu2 mode must be equilibrium, structure, or propagation")
        if int(num_samples) < 1 or int(batch_size) < 1 or int(propagation_steps) < 1:
            raise ValueError("num_samples, batch_size and propagation_steps must be positive")
        if mode == "propagation" and not initial_cif:
            raise ValueError("BioEmu2 propagation requires initial_cif")
        if mode != "propagation" and initial_cif:
            raise ValueError("initial_cif is supported only in propagation mode")
        self.output_dir = pathlib.Path(output_dir).resolve()
        self.python_bin = str(python_bin)
        self.options = dict(cache_dir=str(cache_dir), mode=mode, num_samples=int(num_samples),
                            batch_size=int(batch_size), sampler=sampler, seed=int(seed) if seed is not None else None,
                            initial_cif=str(pathlib.Path(initial_cif).resolve()) if initial_cif else None,
                            lag_time_ns=float(lag_time_ns), propagation_steps=int(propagation_steps),
                            device=device, precision=precision, matmul_precision=matmul_precision)

    def prepare_input(self, inputs, runtime_config):
        requests = {}
        for name, data in inputs.items():
            if not data.chains or any(not isinstance(chain, ProteinChain) for chain in data.chains):
                raise ValueError("BioEmu2 evaluator currently supports protein-only inputs")
            if data.bonded_atom_pairs or data.cyclic_chain_ids or data.cyclic_relative_patches:
                raise ValueError("BioEmu2 evaluator does not support bonded or cyclic inputs")
            chains = []
            for chain in data.chains:
                if chain.ptms:
                    raise ValueError("BioEmu2 evaluator does not support protein modifications")
                if not chain.sequence or not set(chain.sequence) <= set("ACDEFGHIKLMNPQRSTVWY"):
                    raise ValueError("BioEmu2 evaluator requires canonical protein sequences")
                chains.append({"id": chain.id, "sequence": chain.sequence})
            seeds = list(data.rng_seeds or [])
            requests[name] = {"chains": chains, "input_seed": int(seeds[0]) if seeds else None}
        return requests

    def run_function(self, input_feat):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        worker = pathlib.Path(__file__).parent / "wrappers" / "bioemu2_worker.py"
        for name, item in input_feat.items():
            directory = pathlib.Path(tempfile.mkdtemp(prefix="sample_", dir=self.output_dir))
            request = {**self.options, "chains": item["chains"], "output_dir": str(directory)}
            if request["seed"] is None:
                request["seed"] = item["input_seed"]
            request_path = directory / "request.json"
            request_path.write_text(json.dumps(request, indent=2) + "\n")
            log_path = directory / "worker.log"
            with log_path.open("w") as log:
                completed = subprocess.run([self.python_bin, str(worker), str(request_path)],
                                           stdout=log, stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise RuntimeError(f"BioEmu2 failed with exit code {completed.returncode}; see {log_path}")
            results[name] = directory / "representative.cif"
            if not results[name].is_file():
                raise RuntimeError(f"BioEmu2 did not produce {results[name]}; see {log_path}")
        return results

    def prepare_output(self, output, inputs, runtime_config):
        # BioEmu2 does not supply AF-style confidence metrics. Missing keys make
        # confidence-based losses fail rather than inventing pLDDT or pTM.
        return StandardScores(names=list(inputs), plddt={}, complex_paths=output)
